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
from lib.tools.checkdataset import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA_PATH_TRAIN = '/dtu/datasets1/02516/DRIVE/training'
DATA_PATH_TEST = '/zhome/b7/2/219221/IDLCV_Exercise_3_segmentation/dataset/DRIVE/test'

class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        # images_dir 
        # gt_dir 
        # mask FOV
        self.train = train
        if self.train == 'train':
            img_dir = Path(DATA_PATH_TRAIN)
        else:
            img_dir = Path(DATA_PATH_TEST)
            
        self.images = sorted((img_dir/'images').glob('*.tif'))
        self.transform = transform

    def __len__(self):
        'Returns the total number of samples'
        return len(self.images)

    def __getitem__(self, idx):
        'Generates one sample of data'
        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if self.train == 'train': 
            # training set
            mask_path = img_path.parent.parent/'mask'/img_path.name.replace('.tif', '_mask.gif')
            gt_path = img_path.parent.parent/'1st_manual'/img_path.name.replace('_training.tif', '_manual1.gif')
        else:
            # test dataset
            mask_path = img_path.parent.parent/'mask'/img_path.name.replace('.tif', '_mask.gif')
            gt_path = img_path.parent.parent/'labels'/img_path.name.replace('_test.tif', '_manual1.png')
        
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt = (gt>127).astype(np.float32)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask>127).astype(np.float32)

        if self.transform:
                augmented = self.transform(image=img, gt=gt, mask=mask)
                img, gt, mask = augmented['image'], augmented['gt'], augmented['mask']
                
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, gt, mask


def get_train_transform(target_size=(512,512)):
    return A.Compose([
        A.Resize(*target_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.7),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ],
    additional_targets={
        'gt': 'mask',
        'mask': 'mask'
    })

def get_test_transform(target_size=(512,512)):
    return A.Compose([
        A.Resize(*target_size),
        ToTensorV2()
    ],
    additional_targets={
        'gt': 'mask',
        'mask': 'mask'
    })
    
if __name__ == '__main__':
    batch_size = 4
    trainset = DRIVEDataset(train='test', transform=get_test_transform())
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    images, gts, masks = next(iter(train_loader))
    print('images:', images.shape)  # (B,3,H,W)
    print('gts:', gts.shape)        # (B,1,H,W)
    print('masks:', masks.shape)    # (B,1,H,W)
    
    print('\n=== Dtypes ===')
    print('images dtype:', images.dtype)  # should be torch.float32
    print('gts dtype:', gts.dtype)        # should be torch.float32
    print('masks dtype:', masks.dtype)    # should be torch.float32

    print('\n=== Pixel ranges ===')
    print('images min/max:', images.min().item(), images.max().item())  # should be ~0-1
    print('gts unique values:', torch.unique(gts))  # should be 0.0/1.0
    print('masks unique values:', torch.unique(masks))  # should be 0.0/1.0
