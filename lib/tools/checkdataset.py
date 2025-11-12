import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def analyze_dataset(data_loader):
    num_samples = 0
    img_shapes = []
    
    total_foreground = 0
    total_background = 0
    
    for batch in tqdm(data_loader, desc="Analyzing dataset"):
        # 假设 dataset 返回 (image, mask)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, masks = batch[0], batch[1]
        else:
            continue  # 跳过不符合格式的 batch

        bs = images.size(0)
        num_samples += bs

        # image Resolution
        for i in range(bs):
            img_shapes.append(images[i].shape[-2:])  # (H, W)
            
        # mask 稀疏性
        masks = masks.detach().cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]  # 取 channel 维
        elif masks.ndim != 3:
            raise ValueError("Unsupported mask shape: ", masks.shape)
        
        total_foreground += np.sum(masks > 0)   # 认为非 0 是前景
        total_background += np.sum(masks == 0)  # 0 是背景
        
    total_pixels = total_foreground + total_background
    fg_ratio = total_foreground / total_pixels
    bg_ratio = total_background / total_pixels
    
    print("\n=== Dataset Summary ===")
    print(f"Total samples: {num_samples}")
    print(f"Foreground pixels in masks: {total_foreground} ({fg_ratio*100:.2f}%)")
    print(f"Background pixels in masks: {total_background} ({bg_ratio*100:.2f}%)")
    
    # 转为 numpy 方便统计
    img_shapes = np.array(img_shapes)

    if len(img_shapes) > 0:
        h, w = img_shapes[:, 0], img_shapes[:, 1]
        print(f"Image size range: H [{h.min()} - {h.max()}], W [{w.min()} - {w.max()}]")
        print(f"Average image size: ({h.mean():.1f}, {w.mean():.1f})")
