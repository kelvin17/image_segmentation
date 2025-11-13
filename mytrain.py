import os
import numpy as np
import glob

from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from lib.dataset.Ph2Dataset import Ph2
from lib.dataset.DRIVEDataset import *
from lib.model.UNetModel import UNet, UNet2, LightningUNet2
from lib.model.EncDecModel import LightningEncDec
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from lib.losses import *
from measure import *

def train_eval_ph2():
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
    ])

    dataset = Ph2(transform=transform)
    
    # split datasets
    indices = list(range(len(dataset)))
    random_state = 42

    # divide data into 3 splits(60:20:20)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=random_state)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=3)

    # loss - Weight BCE
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pos_weight=compute_pos_weight(train_loader, device)
    # print(f'train pos_weight:{pos_weight}')
    # loss = BCELoss(pos_weight=pos_weight) 
    # loss_name = "WeightedBCE"
    
    # loss = BCELoss() 
    # loss_name = "BCE"
    
    # focal loss
    loss = FocalLoss()
    loss_name = "Focal"
    
    # Metrics
    custom_metrics = {
            "dice": dice_coefficient_withLogtis, 
            "iou": iou_score_withLogtis,
            "pixel_acc": pixel_accuracy_withLogtis, 
            "sensitivity": sensitivity_withLogtis,
            "specificity": specificity_withLogtis
        }
    
    model = LightningUNet2(loss_fn=loss, loss_name=loss_name, metrics=custom_metrics, n_channels=3, n_classes=1, base_c=32)
    
    # model = LightningEncDec(loss_fn=loss, loss_name=loss_name, metrics=custom_metrics, in_channels=3, num_classes=1)
    
    exp_name = f"{model.model_name}-{model.loss_fc_name}"
    
    logger = CSVLogger("logs", name=f"{exp_name}_experiment")
    trainer = Trainer(max_epochs=100, log_every_n_steps=10, accelerator="gpu", logger=logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    model.plot_metrics()
    
def train_eval_DRIVE():
    size = 384
    train_val_dataset = DRIVEDataset(train='train', transform=get_train_transform(target_size=(size,size)))
    # ---------------------------
    # Dataloaders
    # ---------------------------
    # Split train-val: simple 16/4 split = 4/1
    indices = list(range(len(train_val_dataset)))
    random_state = 42

    # divide data into 2 splits(80:20=4:1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=random_state)

    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)
    val_dataset.dataset.transform = get_test_transform(target_size=(size,size))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    test_dataset = DRIVEDataset(train='test', transform=get_test_transform(target_size=(size,size)))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)
    
    custom_metrics = {
            "dice": masked_dice_coef, 
            "iou" : masked_iou_score_withLogtis,
            "pixel_acc": masked_pixel_accuracy_withLogtis, 
            "sensitivity": masked_sensitivity_withLogtis,
            "specificity": masked_specificity_withLogtis
        }
    
    # loss - Weight BCE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pos_weight=compute_pos_weight(train_loader, device)
    # print(f'train pos_weight:{pos_weight}')

    loss = BCELoss(pos_weight=pos_weight, with_mask=True)
    loss_name = 'MaskedWeightedBCE'
    
    # loss - BCE
    # loss = BCELoss(with_mask=True)
    # loss_name = 'MaskedBCE'
    
    # loss - FocalLoss
    # loss = FocalLoss(with_mask=True)
    # loss_name = 'MaskedFocal'
    
    model = LightningUNet2(loss_fn=loss, loss_name=loss_name, metrics=custom_metrics, n_channels=3, n_classes=1, base_c=32, with_mask=True)
    
    # model = LightningEncDec(loss_fn=loss, loss_name=loss_name, metrics=custom_metrics, in_channels=3, num_classes=1, with_mask=True)
    
    exp_name = f"{model.model_name}-{model.loss_fc_name}-DRIVE"
    
    logger = CSVLogger("logs", name=f"{exp_name}_experiment")
    trainer = Trainer(max_epochs=100, log_every_n_steps=10, accelerator="gpu", logger=logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
    model.plot_metrics()
    
    
if __name__ == "__main__":
    print("Begin")
    # train_eval_ph2()
    train_eval_DRIVE()
    print("Finished")