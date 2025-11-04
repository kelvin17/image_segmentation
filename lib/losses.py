import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        up = (y_pred*y_true).mean() + 1
        down = (y_pred+y_true).mean() +1
        
        loss = 1-(up/down)
        return loss
        

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1-pt)**self.gamma * bce
        return loss.mean()


class BCELoss_TotalVariation(nn.Module):
    def __init__(self, tv_weight=0.1):
        super().__init__()
        self.tv_weight = tv_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        # BCE Loss
        bce_loss = self.bce(y_pred, y_true.float())
        
        # TV regularization
        probs = torch.sigmoid(y_pred)  # 转概率
        h_variation = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
        w_variation = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
        tv_loss = h_variation + w_variation
        
        # total loss
        return bce_loss + self.tv_weight * tv_loss

def compute_pos_weight(train_loader, device='cpu'):
    num_foreground = 0
    num_background = 0
    
    for _, masks in train_loader:
        # mask is 0/1，shape = [B,1,H,W]
        if masks.dim() == 4:
            masks = masks.squeeze(1)  # [B,H,W]
        masks = masks.to(device)

        num_foreground += (masks == 1).sum().item()
        num_background += (masks == 0).sum().item()

    pos_weight = num_background / (num_foreground + 1e-6)  # 防止除零
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)