import torch
import os
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from lib.tools.checkdataset import *

DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images'
# DATA_PATH = '/Users/blackbear/Desktop/dtu/semester2/DeepLearningInCV/execise/ph2_data/'

class Ph2(torch.utils.data.Dataset):
    def __init__(self, transform):
        'Initialization'
        self.transform = transform
        # data_path = os.path.join(DATA_PATH, 'train' if train else 'test')
        data_path = DATA_PATH
        self.image_dirs = sorted([os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])


    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_dirs)

    def __getitem__(self, idx):
        single_sample_folder = self.image_dirs[idx]
        image_name = os.path.basename(single_sample_folder)
        
        'image'
        image_dir = os.path.join(single_sample_folder, image_name + "_Dermoscopic_Image/")
        
        files = sorted(os.listdir(image_dir))
        image_path = os.path.join(image_dir, files[0])
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        'lesion'
        lesion_dir = os.path.join(single_sample_folder, image_name + "_lesion/")
        
        files = sorted(os.listdir(lesion_dir))
        lesion_path = os.path.join(lesion_dir, files[0])
        
        lesion = Image.open(lesion_path).convert('L')
        lesion = self.transform(lesion)
        
        'roi'
        # roi_dir = os.path.join(single_sample_folder, image_name + "_roi/")
        # roi_masks = []
        # for item in os.listdir(roi_dir):
        #     roi_item = Image.open(os.path.join(roi_dir, item)).convert('L')
        #     roi_item = self.transform(roi_item)
        #     roi_masks.append(roi_item)
        
        # return image, lesion, roi_item
        return image, lesion
    


if __name__ == '__main__':
    batch_size = 1
    
    # size = 128
    train_transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = Ph2(transform=train_transform)
    print(f'dataset: {len(trainset)}')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            num_workers=1)
    
    analyze_dataset(train_loader)
    
    