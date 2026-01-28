# 加速度场
import os
import sys
import time
import logging
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.tracer import Net as model
from utils.util import set_seed, process_gt, normalize_map, get_optimizer
from utils.viz import viz_pred_test
from utils.evaluation import cal_kl, cal_sim, cal_nss

from models.unet import UNetModelWrapper
from models.refined import RefinerWrapper, ConditionalFlowMatcher

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, help='path to dataset')
parser.add_argument('--save_root', type=str, help='directory to save models and logs')
parser.add_argument('--divide', type=str, default='Seen')
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--iters', type=int, default=20000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=100)
parser.add_argument('--eval_step', type=int, default=2000)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--stage', type=str, default='A', choices=['A','B'], help='Training stage A/B')
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--load_base_ckpt', type=str, default=None, help='Optional: path to base model checkpoint to load at start of Stage B/C')
parser.add_argument('--grad_clip', type=float, default=1.0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.cuda.set_device(0)

time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
save_path = os.path.join(args.save_root, time_str)
os.makedirs(save_path, exist_ok=True)

logging.basicConfig(filename='%s/run.log' % save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

set_seed(args.seed)

from data.fine_agddo15 import TrainData, TestData, SEEN_AFF, UNSEEN_AFF
args.class_names = SEEN_AFF if args.divide == 'Seen' else UNSEEN_AFF

trainset = TrainData(data_root=args.data_root, divide=args.divide, resize_size=args.resize_size, crop_size=args.crop_size)
TrainLoader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
testset = TestData(data_root=args.data_root, divide=args.divide, crop_size=args.crop_size)
TestLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = model(args, 768, 512).to(device)
base_model.train()
optimizer_base, scheduler = get_optimizer(base_model, args)


best_kld = 1000
total_iter = 0
print("Start training - Stage", args.stage)

if args.stage == 'A':
    while True:
        for _, (img, mask, ann) in enumerate(TrainLoader):
            img, mask, ann = img.to(device), mask.to(device), ann.to(device).float()
            pred0, pred1,_, _, loss_dict, loss, _ = base_model(img, mask, label=ann)
            optimizer_base.zero_grad()
            loss.backward()
            optimizer_base.step()
            scheduler.step()

            total_iter += 1

            if total_iter % args.show_step == 0:
                log_str = f"[Stage A] iter {total_iter} | "
                log_str += ' | '.join(['%s: %.4f' % (k, v) for k, v in loss_dict.items()])
                logger.info(log_str)
            
            if total_iter %  args.eval_step == 0:
                base_model.eval()
                GT_path = args.divide + "_AGDDO15sam_gt.t7"
                if not os.path.exists(GT_path):
                    process_gt(args)
                GT_masks = torch.load(GT_path)

                KLs, SIM, NSS = [], [], []
                with torch.no_grad():
                    for step, (image, mask, gt_aff, obj, mask_path) in enumerate(tqdm(TestLoader)):
                        ego_pred0, ego_pred1= base_model(image.cuda(), mask.cuda(),gt_aff=gt_aff)
                        ego_pred = np.array(ego_pred0.squeeze().cpu())
                        ego_pred = normalize_map(ego_pred, args.crop_size)
                        names = mask_path[0].split("/")
                        key = names[-3] + "_" + names[-2] + "_" + names[-1]
                        GT_mask = GT_masks[key] / 255.0
                        GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))
                        kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)
                        KLs.append(kld); SIM.append(sim); NSS.append(nss)

                        if args.viz:
                            if (step + 1) % 40 == 0:
                                img_name = key.split(".")[0]
                                viz_pred_test(save_path, image, ego_pred, GT_mask, args.class_names, gt_aff, img_name, total_iter)

                mKLD, mSIM, mNSS = sum(KLs)/len(KLs), sum(SIM)/len(SIM), sum(NSS)/len(NSS)
                logger.info(f"[Eval] iter {total_iter} | mKLD={mKLD:.3f} mSIM={mSIM:.3f} mNSS={mNSS:.3f}")

                # save best
                if mKLD < best_kld:
                    best_kld = mKLD
                    model_name = 'best_model_' + str(total_iter + 1) + '_' + str(round(best_kld, 3)) \
                                 + '_' + str(round(mSIM, 3)) \
                                 + '_' + str(round(mNSS, 3)) \
                                 + '.pth'
                    torch.save({'iter': total_iter,
                                'model_state_dict': base_model.state_dict(),
                                'optimizer_state_dict': optimizer_base.state_dict()},
                               os.path.join(save_path, f"stageA_best_{total_iter}_{best_kld:.3f}_{mSIM:.3f}_{mNSS:.3f}.pth"))
                base_model.train()

            if total_iter >= args.iters:
                print("Stage A finished")
                exit()

elif args.stage == 'B':
    FM = ConditionalFlowMatcher(sigma=0.0)

    if args.load_base_ckpt is not None and os.path.exists(args.load_base_ckpt):
        ck = torch.load(args.load_base_ckpt, map_location=device)
        base_model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
        logger.info(f"Loaded base checkpoint from {args.load_base_ckpt}")

    a_pred_model = UNetModelWrapper(
        dim=(15, 224, 224),
        num_channels=64,
        num_res_blocks=2,
    ).to(device)

    # freeze base model
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.train()

    optimizera = torch.optim.Adam(a_pred_model.parameters(), lr=2e-7, weight_decay=1e-3)

    warmup_iters = 2000
    total_iters = args.iters

    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        else:
            progress = float(current_step - warmup_iters) / float(max(1, total_iters - warmup_iters))
            import math
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_ = torch.optim.lr_scheduler.LambdaLR(optimizera, lr_lambda)

    best_kld = 1000
    total_iter = 0

    while True:
        for _, (img, mask, gt) in enumerate(TrainLoader):
            img, mask, gt = img.to(device), mask.to(device), gt.to(device).float()
            with torch.no_grad():
                gt = gt.clone()
                if torch.rand(1).item() < 0.5:
                    _, _, _, _, _, _, logits0= base_model(img, mask, label=gt)
                else:
                    dummy_label = torch.zeros_like(gt)
                    _, _, _, _, _, _, logits0= base_model(img, mask, label=dummy_label)

            x0 = logits0
            x1_prob = gt / 255.0
            eps = 1e-6  
            x1 = torch.log((x1_prob + eps) / (1 - x1_prob + eps))
            x1 = torch.clamp(x1, min=-5, max=5)       # min=-10, max=10

            t, xt, v_target = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device); xt = xt.to(device); v_target = v_target.to(device)
            v0 = x1 - xt 
            tau, vtau, a_target = FM.sample_location_and_conditional_flow(v0, v_target)
            tau = tau.to(device); vtau = vtau.to(device); a_target = a_target.to(device)
            a_pred = a_pred_model(tau, vtau, t, xt)
            loss_flow = torch.mean((a_pred - a_target) ** 2)

            loss = loss_flow

            optimizera.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(a_pred_model.parameters(), args.grad_clip)
            optimizera.step()
            scheduler_.step()

            total_iter += 1

            if total_iter % args.show_step == 0:
                logger.info(f"[Stage B] iter {total_iter} | loss={loss.item():.4f}")


            if total_iter % args.eval_step == 0:
                GT_path = args.divide + "_AGDDO15sam_gt.t7"
                if not os.path.exists(GT_path):
                    process_gt(args)
                GT_masks = torch.load(GT_path)

                KLs, SIM, NSS = [], [], []

                a_pred_model.eval()

                with torch.no_grad():
                    for step, (image, mask, gt_aff, obj, mask_path) in enumerate(tqdm(TestLoader)):
                        image, mask = image.to(device), mask.to(device)
                        base_model.train()
                        b, _, h, w = image.shape
                        dummy_label = torch.zeros(b, 15, h, w, device=device)
                        _, _, _, _, _, _, logits0 = base_model(image, mask, label=dummy_label, return_logits=True)
                        base_model.eval()

                        xt = logits0.clone()
                        vtau = torch.randn_like(xt, device=device) * 0.1
                        batchsize = xt.shape[0]
                        
                        N = 2   #2        # 10 
                        M = 2   #2        # 20 
                        
                        t_values = torch.arange(N, device=device) / N
                        tau_values = torch.arange(M, device=device) / M

                        for i in range(N):
                            t = t_values[i].expand(batchsize)
                            for j in range(M):
                                tau = tau_values[j].expand(batchsize) 
                                a = a_pred_model(tau, vtau, t, xt)
                                vtau += a / M
                            xt += vtau / N

                        temperature = 1.0          # 5.0
                        xt_scaled = xt / temperature

                        final_pred_prob = torch.sigmoid(xt_scaled)

                        b_final, c_final, h_final, w_final = final_pred_prob.shape
                        if gt_aff is not None:
                            out = torch.zeros(b_final, h_final, w_final).cuda()
                            for b_ in range(b_final):
                                out[b_] = final_pred_prob[b_, gt_aff[b_]]
                            ego_pred = np.array(out.squeeze().cpu())
                        else:
                            ego_pred = np.array(final_pred_prob.squeeze().cpu())

                        names = mask_path[0].split("/")
                        key = names[-3] + "_" + names[-2] + "_" + names[-1]
                        GT_mask = GT_masks[key] / 255.0
                        GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))

                        kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)

                        KLs.append(kld)
                        SIM.append(sim)
                        NSS.append(nss)                    

                        if args.viz and (step + 1) % 40 == 0:
                            img_name = key.split(".")[0]
                            viz_pred_test(save_path, image, ego_pred, GT_mask, args.class_names, gt_aff, img_name, total_iter)

                mKLD, mSIM, mNSS = sum(KLs)/len(KLs), sum(SIM)/len(SIM), sum(NSS)/len(NSS)
                logger.info(f"[Eval] iter {total_iter} | mKLD={mKLD:.3f} mSIM={mSIM:.3f} mNSS={mNSS:.3f}")

                if mKLD < best_kld:
                    best_kld = mKLD
                    save_name = f"stageB_best_{total_iter}_{best_kld:.3f}_{mSIM:.3f}_{mNSS:.3f}.pth"
                    torch.save({
                        'iter': total_iter,
                        'model_state_dict': a_pred_model.state_dict(),
                        'optimizer_state_dict': optimizera.state_dict()
                    }, os.path.join(save_path, save_name))
                    logger.info(f"Saved best refiner model: {save_name}")

                base_model.train()

            if total_iter >= args.iters:
                print("Stage B finished")
                exit()
