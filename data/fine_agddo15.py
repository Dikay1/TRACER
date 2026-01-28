import os
from os.path import join as opj
import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF

import cv2
from pathlib import Path

BASE_OBJ = ['bag', 'cloth_hat', 'cloth_sleeveless', 'curtains', 'dress', 'hat', 'long_T-shirt', 
            'mask', 'pants', 'short_T-shirt', 'shorts', 'socks', 'tie', 'tissue', 'towel']
SEEN_AFF = ['clip', 'grasp_hat', 'grasp_pantswaist', 'grasp_shoulder', 'grasp_sleeve', 'grasp_strap',
            'hang', 'pick',  'place', 'pull', 'pull_out', 'put_center', 'put_hem', 'put_pantslegs', 'roll']
UNSEEN_AFF = []

class TrainData(data.Dataset):
    def __init__(self, data_root, divide, resize_size=256, crop_size=224):

        self.data_root = Path(data_root)
        divide_path = 'one-shot-' + divide.lower()
        self.train_path = self.data_root / divide_path

        self.img_ann_list = []
        for file in os.listdir(self.train_path):
            if file.endswith('.jpg'):
                img_path = self.train_path / file
                sam_path = self.train_path / f"{Path(file).stem}_sam.png"
                ann_path = self.train_path / file.replace('.jpg', '.npy')
                if sam_path.exists():
                    self.img_ann_list.append([
                        str(img_path),
                        str(sam_path),
                        str(ann_path)
                    ])

        num_obj = 15 if divide == 'Seen' else 7
        assert len(self.img_ann_list) == num_obj, "Each object should only provide one sample"

        self.resize_size = resize_size
        self.crop_size = crop_size

        self.orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8)

    # def extract_orb_keypoints(self, image_tensor):
    #     image_np = image_tensor.permute(1, 2, 0).numpy() * 255
    #     image_np = image_np.astype(np.uint8)
    #     gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    #     kp = self.orb.detect(gray, None)

    #     kp_coords = torch.tensor([p.pt for p in kp], dtype=torch.float32)  # (N, 2)
    #     return kp_coords

    def __getitem__(self, item):
        # img, ann = self.img_ann_list[item]
        img_path, sam_path, ann_path = self.img_ann_list[item]

        # 读取数据
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # [H,W,3]
        sam_mask = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)          # [H,W]
        ann = np.load(ann_path)                                        # 标签

        img_pil = Image.open(img_path).convert("RGB")

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        sam_mask_tensor = torch.from_numpy(sam_mask).float().unsqueeze(0) / 255.0

        masked_image_tensor = image_tensor * sam_mask_tensor

        # 转 PIL 用于 transform
        masked_np = (masked_image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        masked_pil = Image.fromarray(masked_np)
        ann_tensor = torch.from_numpy(ann)

        if self.transform is not None:
            img_pil, masked_pil, ann_tensor = self.transform(img_pil, masked_pil, ann_tensor)

        # img = Image.open(opj(self.train_path, img_path)).convert('RGB')
        # ann = torch.from_numpy(np.load(opj(self.train_path, ann)))
        # img, ann = self.transform(img, ann)

        # keypoints = self.extract_orb_keypoints(masked_image_tensor)
 
        return img_pil, masked_pil, ann_tensor

    def transform(self, img, mask, ann):
        resize = transforms.Resize(size=(self.resize_size, self.resize_size), antialias=None)
        img, mask, ann = resize(img), resize(mask), resize(ann)

        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        ann = TF.crop(ann, i, j, h, w)

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            ann = TF.hflip(ann)

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img = TF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask = TF.normalize(mask, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return img, mask, ann

    def __len__(self):
        return len(self.img_ann_list)


class TestData(data.Dataset):
    def __init__(self, data_root, divide, crop_size=224):

        self.data_root = opj(data_root, divide, 'testset')
        self.ego_path = opj(self.data_root, 'egocentric')
        self.mask_path = opj(self.data_root, 'GT')
        self.divide = divide

        self.image_list = []
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        
        self.orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8)

        files = os.listdir(self.ego_path)
        for file in files:
            file_path = os.path.join(self.ego_path, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    sam_path = os.path.join(obj_file_path, img.replace(".jpg", "_sam.png"))
                    label_path = os.path.join(self.mask_path, file, obj_file, img[:-3] + "png")

                    if os.path.exists(sam_path) and os.path.exists(label_path):
                        self.image_list.append((img_path, sam_path))
                    # if os.path.exists(mask_path):
                    #     self.image_list.append(img_path)

    # def extract_orb_keypoints(self, image_tensor):
    #     image_np = image_tensor.permute(1, 2, 0).numpy() * 255
    #     image_np = image_np.astype(np.uint8)
    #     gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    #     kp = self.orb.detect(gray, None)

    #     kp_coords = torch.tensor([p.pt for p in kp], dtype=torch.float32)  # (N, 2)
    #     return kp_coords

    def __getitem__(self, item):
        img_path, sam_path = self.image_list[item]

        img, masked_img = self.load_img(img_path, sam_path)

        # # 提取 ORB keypoints
        # keypoints = self.extract_orb_keypoints(masked_img)

        # image_path = self.image_list[item]
        names = img_path.split("/")
        aff_name, object = names[-3], names[-2]

        # image = self.load_img(img_path)
        # keypoints = self.extract_orb_keypoints(image)

        AFF_CLASS = SEEN_AFF if self.divide == 'Seen' else UNSEEN_AFF
        gt_aff = AFF_CLASS.index(aff_name)
        names = img_path.split("/")
        mask_path = os.path.join(self.mask_path, names[-3], names[-2], names[-1][:-3] + "png")

        return img, masked_img, gt_aff, object, mask_path

    # def load_img(self, path):
    #     img = Image.open(path).convert('RGB')
    #     img = self.transform(img)
    #     return img

    def load_img(self, img_path, sam_path):
        img = Image.open(img_path).convert('RGB')
        sam_mask = Image.open(sam_path).convert('L')
        img = self.transform(img)
        mask_tensor = self.mask_transform(sam_mask)
        # 转 tensor
        # img_tensor = transforms.ToTensor()(img)
        # mask_tensor = transforms.ToTensor()(sam_mask)
        masked_img = img * mask_tensor
        # 转 PIL 做 transform
        # masked_pil = transforms.ToPILImage()(masked_tensor)
        # masked_img = self.transform(masked_pil)
        return img, masked_img
    
    def __len__(self):

        return len(self.image_list)