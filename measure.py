# Here, you can load the predicted segmentation masks, and evaluate the
# performance metrics (accuracy, etc.)
import torch

def dice_coefficient_withLogtis(y_pred, y_true, eps=1e-7):
    y_pred = torch.sigmoid(y_pred)
    # y_pred = (y_pred > 0.5).float()
    y_pred = y_pred.float()
    y_true = y_true.float()
    intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    
    print('\n'+'*'*20 + 'mask value evaluation' + '*'*20)
    print(f'target: min={y_true.min()}, max={y_true.max()}, mean={y_true.mean()}')
    print(f'predic: min={y_pred.min()}, max={y_pred.max()}, mean={y_pred.mean()}')
    print('\n'+'*'*50)
    return dice.mean()

def iou_score_withLogtis(y_pred, y_true, eps=1e-7):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def pixel_accuracy_withLogtis(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    correct = (y_pred == y_true).float().sum(dim=(1, 2, 3))
    # (y_pred == y_true) 是逐像素比较的，比如y_pred = [0,1,0], y_true = [1,1,0] 那么结果是[False, True, True]
    total = torch.numel(y_true[0])
    return (correct / total).mean()

def sensitivity_withLogtis(y_pred, y_true, eps=1e-7):
    # 在所有真实病变中，有多少被模型正确预测出来 - False Negative错识别为健康的，它真实是病变的
    # True Positive / (True Positive + False Negative)
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    tp = (y_pred * y_true).sum(dim=(1, 2, 3))
    fn = ((1 - y_pred) * y_true).sum(dim=(1, 2, 3))
    sens = (tp + eps) / (tp + fn + eps)
    return sens.mean()

def specificity_withLogtis(y_pred, y_true, eps=1e-7):
    # 在所有健康的案例，有多少被模型正确预测出来 - False Positive错识别为病例的，它真实是健康的
    # True Negative / (True Negative + False Positive)
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    tn = ((1 - y_pred) * (1 - y_true)).sum(dim=(1, 2, 3))
    fp = (y_pred * (1 - y_true)).sum(dim=(1, 2, 3))
    spec = (tn + eps) / (tn + fp + eps)
    return spec.mean()

# ---------------------------
# Dice metric (masked)
# ---------------------------
def masked_dice_coef(logits, targets, mask, eps=1e-6):
    probs = torch.sigmoid(logits)
    # pred = (probs>0.5).float()
    pred = probs.float()
    targets = targets.float()
    mask = mask.float()
    
    intersection = (pred * targets * mask).sum(dim=(1,2,3))
    union = (pred*mask).sum(dim=(1,2,3)) + (targets*mask).sum(dim=(1,2,3))
    dice = (2*intersection + eps)/(union + eps)
    
    print('\n'+'*'*20 + 'mask value evaluation' + '*'*20)
    print(f'mask: min={mask.min()}, max={mask.max()}, mean={mask.mean()}')
    print(f'target: min={targets.min()}, max={targets.max()}, mean={targets.mean()}')
    print(f'predic: min={pred.min()}, max={pred.max()}, mean={pred.mean()}')
    print(f'logit: min={logits.min()}, max={logits.max()}, mean={logits.mean()}')
    print('\n'+'*'*50)

    return dice.mean()

def masked_iou_score_withLogtis(y_pred, y_true, mask, eps=1e-7):
    y_pred = torch.sigmoid(y_pred)
    # y_pred = (y_pred > 0.5).float()
    y_pred = y_pred.float()
    y_true = y_true.float()
    mask = mask.float()
    intersection = (y_pred * y_true * mask).sum(dim=(1, 2, 3))
    union = (y_pred*mask).sum(dim=(1, 2, 3)) + (y_true*mask).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def masked_pixel_accuracy_withLogtis(y_pred, y_true, mask):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    # correct = ((y_pred*mask) == (y_true*mask)).float().sum(dim=(1, 2, 3)) -- 这么写会让mask之外的部分被计入正确的值
    correct = ((y_pred == y_true).float() * mask).sum(dim=(1, 2, 3))
    # (y_pred == y_true) 是逐像素比较的，比如y_pred = [0,1,0], y_true = [1,1,0] 那么结果是[False, True, True]
    total = torch.numel(y_true[0])
    return (correct / total).mean()


def masked_sensitivity_withLogtis(y_pred, y_true, mask, eps=1e-7):
    # 在所有真实病变中，有多少被模型正确预测出来 - False Negative错识别为健康的，它真实是病变的
    # True Positive / (True Positive + False Negative)
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    tp = (y_pred * y_true * mask).sum(dim=(1, 2, 3))
    fn = ((1 - y_pred) * y_true * mask).sum(dim=(1, 2, 3))
    sens = (tp + eps) / (tp + fn + eps)
    return sens.mean()

def masked_specificity_withLogtis(y_pred, y_true, mask, eps=1e-7):
    # 在所有健康的案例，有多少被模型正确预测出来 - False Positive错识别为病例的，它真实是健康的
    # True Negative / (True Negative + False Positive)
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    tn = ((1 - y_pred) * (1 - y_true) * mask).sum(dim=(1, 2, 3))
    fp = (y_pred * (1 - y_true) * mask).sum(dim=(1, 2, 3))
    spec = (tn + eps) / (tn + fp + eps)
    return spec.mean()