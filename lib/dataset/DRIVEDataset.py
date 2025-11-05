import torch
import os
import glob
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

DATA_PATH = '/dtu/datasets1/02516/DRIVE'

class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        # images_dir 
        # gt_dir 
        # mask FOV
        self.train = train
        if self.train == 'train':
            img_dir = Path(DATA_PATH) / 'training'
        else:
            img_dir = Path(DATA_PATH) / 'test'
        
        print(img_dir)
        self.images = sorted(img_dir.glob('images/*.tif'))

        self.transform = transform

    def __len__(self):
        'Returns the total number of samples'
        return len(self.images)

    def __getitem__(self, idx):
        'Generates one sample of data'
        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = torch.from_numpy(img).permute(2,0,1)  # C,H,W
        
        if self.train == 'train': 
            # training set
            gt_path = img_path.name.replace('_training.tif', '_manual1.gif')
            gt_path = img_path.parent.parent / '1st_manual' / gt_path  # 上两级目录再拼接

            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = (gt>127).astype(np.float32)
            gt = torch.from_numpy(gt).unsqueeze(0)
            
            mask_path = img_path.parent.parent / 'mask' / img_path.name.replace('.tif', '_mask.gif')
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask>127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
            
            if self.transform:
                img, gt, mask = self.transform(img, gt, mask)
            
            return img, gt, mask
        else:
            # test, no gt
            mask_path = img_path.parent.parent / 'mask' / img_path.name.replace('.tif', '_mask.gif')
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask>127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
            
            if self.transform:
                img, mask = self.transform(img, mask)
            
            return img, mask


class JointTransform:
    def __init__(self, resize=(512, 512), augment=True):
        """
        resize: tuple (H, W) to resize all images and masks to same size
        augment: whether to apply random flips
        """
        self.resize = resize
        self.augment = augment

    def __call__(self, img, mask, gt=None):
        # resize — 注意 interpolation
        img = TF.resize(img, self.resize, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.resize, interpolation=TF.InterpolationMode.NEAREST)
        if gt is not None:
            gt = TF.resize(gt, self.resize, interpolation=TF.InterpolationMode.NEAREST)

        # augment
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
                if gt is not None: gt = TF.hflip(gt)
            if random.random() > 0.5:
                img = TF.vflip(img)
                if gt is not None: gt = TF.vflip(gt)
                mask = TF.vflip(mask)

        # To tensor
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
        if not isinstance(mask, torch.Tensor):
            mask = TF.to_tensor(mask)
        if gt is not None and not isinstance(gt, torch.Tensor):
            gt = TF.to_tensor(gt)
            
        return (img, mask, gt) if gt is not None else (img, mask)
    
if __name__ == '__main__':
    batch_size = 1
    
    size = 384
    transform = JointTransform(resize=(size, size))
    trainset = DRIVEDataset(train='train', transform=transform)
    print(f'trainset: {len(trainset)}')
    
    indices = list(range(len(trainset)))
    random_state = 42

    # divide data into 2 splits(80:20=4:1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=random_state)

    train_dataset = Subset(trainset, train_idx)
    val_dataset = Subset(trainset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)
    
    image, gt, mask = next(iter(train_loader))
    
    print(image.shape)
    print(gt.shape)
    print(mask.shape)
    
    print('*'*50)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)
    image, gt, mask = next(iter(val_loader))
    
    print(image.shape)
    print(gt.shape)
    print(mask.shape)