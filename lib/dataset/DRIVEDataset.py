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
            img_dir = DATA_PATH_TRAIN
        else:
            img_dir = DATA_PATH_TEST
        
        self.images = sorted(img_dir.glob('images/*.tif'))
        self.transform = transform

    def __len__(self):
        'Returns the total number of samples'
        return len(self.images)

    def __getitem__(self, idx):
        'Generates one sample of data'
        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        # img = torch.from_numpy(img).permute(2,0,1)  # C,H,W
        
        if self.train == 'train': 
            # training set
            gt_path = img_path.name.replace('_training.tif', '_manual1.gif')
            gt_path = img_path.parent.parent / '1st_manual' / gt_path  # 上两级目录再拼接

            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = (gt>127).astype(np.float32)
            # gt = torch.from_numpy(gt).unsqueeze(0)
            # gt = np.expand_dims(gt, axis=-1)
            
            mask_path = img_path.parent.parent / 'mask' / img_path.name.replace('.tif', '_mask.gif')
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask>127).astype(np.float32)
            
            if self.transform:
                augmented = self.transform(image=img, gt=gt, mask=mask)
                img, gt, mask = augmented['image'], augmented['gt'], augmented['mask']
                
            # gt = gt.squeeze(-1)
            # mask = mask.squeeze(-1)
            mask = mask.unsqueeze(0)
            gt = gt.unsqueeze(0)
            
            
            return img, gt, mask
        else:
            # test, gt
            mask_path = img_path.parent.parent / 'mask' / img_path.name.replace('.tif', '_mask.gif')
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask>127).astype(np.float32)
            
            gt_path = img_path.parent.parent / 'labels' / img_path.name.replace('_test.tif', '_manual1.png')
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = (gt>127).astype(np.float32)
            
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
                
            mask = mask.unsqueeze(0)
            
            return img, mask


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
    
# if __name__ == '__main__':
    # batch_size = 1
    
    # trainset = DRIVEDataset(train='train', transform=transform)
    
    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
    #                         num_workers=1)
    
    # analyze_dataset(train_loader)