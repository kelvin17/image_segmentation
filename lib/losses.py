import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Masked BCE Loss
# ---------------------------
class BCELoss(nn.Module):
    def __init__(self, pos_weight=None, with_mask=False):
        super().__init__()
        self.with_mask = with_mask
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true, mask=None):
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=self.pos_weight, reduction='none')
        if self.with_mask:
            loss = loss * mask
            denom = (mask.sum() + 1e-6)
            loss = loss.sum()/denom
        else:
            loss = loss.mean()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, with_mask=False, smooth=1e-7):
        super().__init__()
        self.with_mask = with_mask
        self.smooth = smooth

    def forward(self, y_pred, y_true, mask=None):
        y_pred = torch.sigmoid(y_pred)
        
        # flatten all
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if self.with_mask and mask is not None:
            mask = mask.float().view(-1)
            intersection = (y_pred * y_true * mask).sum()
            union = (y_pred * mask).sum() + (y_true * mask).sum()
        else:
            intersection = (y_pred * y_true).sum()
            union = y_pred.sum() + y_true.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, with_mask=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.with_mask = with_mask

    def forward(self, logits, targets, mask=None, eps=1e-7):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1-pt)**self.gamma * bce
        
        if self.with_mask and mask is not None:
            mask = mask.float()
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + eps)
        else:
            loss = loss.mean()
            
        return loss


class BCELoss_TotalVariation(nn.Module):
    def __init__(self, tv_weight=0.1, with_mask=False):
        super().__init__()
        self.tv_weight = tv_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.with_mask = with_mask

    def forward(self, y_pred, y_true):
        # BCE Loss
        bce_loss = self.bce(y_pred, y_true.float())
        
        # TV regularization
        probs = torch.sigmoid(y_pred)  # 转概率
        h_variation = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
        w_variation = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
        
        if self.with_mask and mask is not None:
            mask_h = mask[:, :, 1:, :] * mask[:, :, :-1, :] 
            mask_w = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            h_loss = (h_variation * mask_h).sum() / (mask_h.sum() + eps)
            w_loss = (w_variation * mask_w).sum() / (mask_w.sum() + eps)
            tv_loss = h_loss + w_loss
        else:
            tv_loss = h_variation + w_variation
        
        # total loss
        return bce_loss + self.tv_weight * tv_loss

def compute_pos_weight(train_loader, device='cpu'):
    num_foreground = 0
    num_background = 0
    
    for batch in train_loader:
        if len(batch) == 2:
            _, labels = batch
        elif len(batch) == 3:
            _, labels, _ = batch
        # labels is 0/1，shape = [B,1,H,W]
        if labels.dim() == 4:
            labels = labels.squeeze(1)  # [B,H,W]
        labels = labels.to(device)

        num_foreground += (labels == 1).sum().item()
        num_background += (labels == 0).sum().item()

    pos_weight = num_background / (num_foreground + 1e-6)  # 防止除零
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)