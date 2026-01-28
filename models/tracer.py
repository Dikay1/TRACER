import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import numpy as np

from models.clip import clip
from models.cocoop import TextEncoder, PromptLearner
from models.seg_decoder import SegDecoder
from models.SEM import Semantic_Enhancement_Module

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def sobel_operator(input_tensor):
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=input_tensor.device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32, device=input_tensor.device)
    channels = input_tensor.size(1)
    kernel_x = kernel_x.view(1, 1, 3, 3).expand(channels, -1, -1, -1)
    kernel_y = kernel_y.view(1, 1, 3, 3).expand(channels, -1, -1, -1)
    grad_x = F.conv2d(input_tensor, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(input_tensor, kernel_y, padding=1, groups=channels)
    return grad_x, grad_y

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

proj_img = nn.Linear(512, 512).cuda()
proj_txt = nn.Linear(512, 512).cuda()

class Net(nn.Module):
    """
    Base network for heatmap prediction.
    Supports forward(..., return_feats=True) to return features for refiner.
    Training behavior unchanged when return_feats=False.
    """
    def __init__(self, args, input_dim, out_dim, dino_pretrained='dinov2_vitb14'):
        super().__init__()
        self.dino_pretrained = dino_pretrained
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.class_names = args.class_names
        self.num_aff = len(self.class_names)

        self.embedder = Mlp(in_features=input_dim, hidden_features=int(out_dim), out_features=out_dim, act_layer=nn.GELU, drop=0.)
        self.dino_model = torch.hub.load(repo_or_dir='./models/dinov2', model='dinov2_vitb14', source='local', force_reload=False).cuda()

        clip_model = load_clip_to_cpu('ViT-B/16').float()
        classnames = [a.replace('_', ' ') for a in self.class_names]
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.aff_text_encoder = TextEncoder(clip_model)

        self.seg_decoder = SegDecoder(embed_dims=out_dim, num_layers=2)
        self.merge_weight = nn.Parameter(torch.zeros(3))

        self.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
        self.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)])

        self.lln_norm_1 = nn.LayerNorm(out_dim)
        self.lln_norm_2 = nn.LayerNorm(out_dim)
        self.linear_cls = nn.Linear(input_dim, out_dim)
        self.sem = Semantic_Enhancement_Module(num_class=self.num_aff, groups=15)

        self._freeze_stages(exclude_key=['embedder', 'ctx', 'seg_decoder', 'lln_', 'merge_weight', 'linear_cls'])
        self.step = 0

    def forward(self, img, mask, keypoints=None, label=None, gt_aff=None, step=None, return_feats=False, return_logits=False):
        device = img.device
        b, _, h, w = img.shape
        b, _, h, w = mask.shape

        dino_out0 = self.dino_model.get_intermediate_layers(img, n=3, return_class_token=True)
        dino_out1 = self.dino_model.get_intermediate_layers(mask, n=3, return_class_token=True)
        merge_weight = torch.softmax(self.merge_weight, dim=0)

        dino_dense0 = 0
        dino_dense1 = 0

        for i, (feat0, feat1) in enumerate(zip(dino_out0, dino_out1)):
            feat0_ = self.lln_linear[i](feat0[0])
            feat0_ = self.lln_norm[i](feat0_)
            dino_dense0 += feat0_ * merge_weight[i]

            feat1_ = self.lln_linear[i](feat1[0])
            feat1_ = self.lln_norm[i](feat1_)
            dino_dense1 += feat1_ * merge_weight[i]

        dino_dense0 = self.lln_norm_1(self.embedder(dino_dense0))
        dino_dense1 = self.lln_norm_1(self.embedder(dino_dense1))

        dino_dense0 = self.sem(dino_dense0, label)  # [1,256,512]
        dino_dense1 = self.sem(dino_dense1, label)

        prompts0 = self.prompt_learner(dino_dense0).squeeze(0)
        prompts1 = self.prompt_learner(dino_dense1).squeeze(0)
        tokenized_prompts = self.tokenized_prompts

        text_features0 = self.lln_norm_2(self.aff_text_encoder(prompts0, tokenized_prompts))
        text_features1 = self.lln_norm_2(self.aff_text_encoder(prompts1, tokenized_prompts))

        dino_cls0 = dino_out0[-1][1]
        dino_cls0 = self.linear_cls(dino_cls0)
        dino_cls1 = dino_out1[-1][1]
        dino_cls1 = self.linear_cls(dino_cls1)

        text_features0 = text_features0.unsqueeze(0).expand(b, -1, -1)
        text_features1 = text_features1.unsqueeze(0).expand(b, -1, -1)

        text_features0_out, attn_out0, _ = self.seg_decoder(text_features0, dino_dense0, dino_cls0)
        text_features1_out, attn_out1, _ = self.seg_decoder(text_features1, dino_dense1, dino_cls1)

        attn0 = (text_features0_out[-1] @ dino_dense0.transpose(-2, -1)) * (512**-0.5)
        attn_out0 = torch.sigmoid(attn0)
        attn_out0 = attn_out0.reshape(b, -1, h // 14, w // 14)
        pred0 = F.interpolate(attn_out0, img.shape[-2:], mode='bilinear', align_corners=False)

        attn1 = (text_features1_out[-1] @ dino_dense1.transpose(-2, -1)) * (512**-0.5)
        attn_out1 = torch.sigmoid(attn1)
        attn_out1 = attn_out1.reshape(b, -1, h // 14, w // 14)
        pred1 = F.interpolate(attn_out1, mask.shape[-2:], mode='bilinear', align_corners=False)

        attn0_reshaped = attn0.reshape(b, -1, h // 14, w // 14)
        logits0 = F.interpolate(attn0_reshaped, img.shape[-2:], mode='bilinear', align_corners=False)

        if self.training:

            assert label is not None, 'Label should be provided during training'
            loss_bce0 = nn.BCELoss()(pred0, label / 255.0)
            loss_bce1 = nn.BCELoss()(pred1, label / 255.0)
            loss_bce = loss_bce0 + loss_bce1

            with torch.no_grad():
                bce_scale = loss_bce.item()
                eps = 1e-8
                temp = 1.0
                w_consistency = 0.1
                w_grad = 0.05
                dynamic_w_consistency = w_consistency * temp / (bce_scale + eps)
                dynamic_w_grad = w_grad * temp / (bce_scale + eps)

            p0 = torch.sigmoid(pred0)
            p1 = torch.sigmoid(pred1)

            kl_01 = torch.sum(p0 * torch.log((p0 + eps) / (p1 + eps)), dim=1).mean()
            kl_10 = torch.sum(p1 * torch.log((p1 + eps) / (p0 + eps)), dim=1).mean()
            loss_consistency = 0.5 * (kl_01 + kl_10)

            grad_x, grad_y = sobel_operator(pred0)
            label_grad_x, label_grad_y = sobel_operator(label.float())
            label_boundary = (torch.abs(label_grad_x) + torch.abs(label_grad_y)) > 0.1

            grad_penalty = torch.mean(
                (grad_x.abs() * label_boundary.float())**2 +
                (grad_y.abs() * label_boundary.float())**2
            )

            loss = loss_bce + dynamic_w_consistency * loss_consistency + dynamic_w_grad * grad_penalty

            loss_dict = {
                'loss_bce0': loss_bce0,
                'loss_bce1': loss_bce1,
                'loss_consistency': loss_consistency,
                'grad_penalty': grad_penalty,
                'loss': loss
            }

            return pred0, pred1, dino_dense0, text_features0, loss_dict, loss, logits0

        else:
            if return_logits:
                if gt_aff is not None:
                    out_logits0 = torch.zeros(b, h, w).cuda()
                    for b_ in range(b):
                        out_logits0[b_] = logits0[b_, gt_aff[b_]]
                    return out_logits0

            else:
                if gt_aff is not None:
                    out0 = torch.zeros(b, h, w).cuda()
                    out1 = torch.zeros(b, h, w).cuda()
                    for b_ in range(b):
                        out0[b_] = pred0[b_, gt_aff[b_]]
                        out1[b_] = pred1[b_, gt_aff[b_]]
                    return out0, out1

    def _freeze_stages(self, exclude_key=None):
        for n, m in self.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    raise AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False