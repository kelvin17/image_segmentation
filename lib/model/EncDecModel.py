import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

class EncDec(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        # print(f'after pool0:{e0.shape}')
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        # print(f'after pool1:{e1.shape}')
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        # print(f'after pool2:{e2.shape}')
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))
        # print(f'after pool3:{e3.shape}')

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))
        # print(f'after bottleneck_conv:{b.shape}')

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        # print(f'after dec_conv0:{d0.shape}')
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        # print(f'after dec_conv1:{d1.shape}')
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        # print(f'after dec_conv2:{d2.shape}')
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        # print(f'after dec_conv3:{d3.shape}')
        return d3


class LightningEncDec(LightningModule):
    def __init__(self, loss_fn=nn.BCEWithLogitsLoss(), loss_name=None, metrics=None,
                 in_channels=3, num_classes=1, with_mask=False):
        super().__init__()
        self.model = EncDec()
        self.criterium = loss_fn
        self.model_name = "EncDec"
        self.with_mask = with_mask
        self.loss_fc_name = (loss_name if loss_name is not None else loss_fn.__class__.__name__)
        
        self.metrics = metrics

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val": {k: [] for k in self.metrics.keys()},
            "test": {k: [] for k in self.metrics.keys()}
        }
        
        self.train_epoch_outputs = []
        self.val_epoch_outputs = []
        self.test_epoch_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def compute_metrics(self, y_hat, y_true, soft=True):
        metric_logs = {}
        for name, func in self.metrics.items():
            val = func(y_hat, y_true, soft)
            if isinstance(val, torch.Tensor):
                val = val.item()
            metric_logs[name] = val
        return metric_logs
    
    def compute_metrics_mask(self, y_hat, y_true, mask, soft=True):
        metric_logs = {}
        for name, func in self.metrics.items():
            val = func(y_hat, y_true, mask, soft)
            if isinstance(val, torch.Tensor):
                val = val.item()
            metric_logs[name] = val
        return metric_logs
    
    def training_step(self, batch, batch_idx):
        if self.with_mask:
            images, labels, mask = batch
            outputs = self(images)
            loss = self.criterium(outputs, labels, mask)
        else:
            images, labels = batch
            outputs = self(images)
            loss = self.criterium(outputs, labels)
            
        self.train_epoch_outputs.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        if self.train_epoch_outputs:
            avg_loss = torch.stack(self.train_epoch_outputs).mean().item()
            self.history["train_loss"].append(avg_loss)
            self.train_epoch_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        if self.with_mask:
            images, labels, masks = batch
            outputs = self(images)
            loss = self.criterium(outputs, labels, masks)
            metrics = self.compute_metrics_mask(outputs, labels, masks)
        else:
            images, labels = batch
            outputs = self(images)
            loss = self.criterium(outputs, labels)
            metrics = self.compute_metrics(outputs, labels)
        
        self.val_epoch_outputs.append({"val_loss": loss.detach(), **metrics})
        
        return {"val_loss": loss.detach(), **metrics}
        
    def on_validation_epoch_end(self):
        if self.val_epoch_outputs:
            avg_loss = torch.stack([x["val_loss"] for x in self.val_epoch_outputs]).mean().item()
            self.history["val_loss"].append(avg_loss)
            for k in self.metrics.keys():
                avg_metric = torch.tensor([x[k] for x in self.val_epoch_outputs]).mean().item()
                self.history["val"][k].append(avg_metric)
            # log metrics
            log_dict = {"val_loss": avg_loss}
            log_dict.update({f"val_{k}": self.history["val"][k][-1] for k in self.metrics.keys()})
            self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.val_epoch_outputs.clear()

    def test_step(self, batch, batch_idx):
        if self.with_mask:    
            images, labels, masks = batch
            outputs = self(images)
            metrics = self.compute_metrics_mask(outputs, labels, masks, soft=False)
        else:
            images, labels = batch
            outputs = self(images)
            metrics = self.compute_metrics(outputs, labels, soft=False)
        
        self.test_epoch_outputs.append(metrics)
        return metrics
    
    def on_test_epoch_end(self):
        if self.test_epoch_outputs:
            for k in self.metrics.keys():
                avg_metric = torch.tensor([x[k] for x in self.test_epoch_outputs]).mean().item()
                self.history["test"][k].append(avg_metric)
            # log metrics
            log_dict = {f"test_{k}": self.history["test"][k][-1] for k in self.metrics.keys()}
            self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.test_epoch_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def plot_metrics(self):
        def smooth_curve(y, sigma=2):
            return gaussian_filter1d(y, sigma=sigma) if len(y) > 3 else y

        # --- 1. Loss ---
        plt.figure(figsize=(8,5))
        if self.history["train_loss"]:
            plt.plot(smooth_curve(self.history["train_loss"]), label="Train Loss", color="blue")
        if self.history["val_loss"]:
            plt.plot(smooth_curve(self.history["val_loss"]), label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss on {self.model_name} with {self.loss_fc_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{self.model_name}-{self.loss_fc_name}-loss_curve.png", dpi=300)
        plt.close()

        # --- 2. Metrics （Val every epoch + Test point） ---
        plt.figure(figsize=(10,6))
        metrics = list(self.metrics.keys())
        colors = ["r", "g", "b", "m", "c"]

        for i, m in enumerate(metrics):
            # val curve
            if self.history["val"][m]:
                plt.plot(smooth_curve(self.history["val"][m]), "--", color=colors[i], label=f"Val {m}")
            
            # test point
            if self.history["test"][m]:
                # x of last epoch 
                last_epoch = len(self.history["val"][m]) - 1
                plt.plot(last_epoch, self.history["test"][m][-1], "o", color=colors[i], markersize=8, label=f"Test {m}")

        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title(f"Metrics on {self.model_name} with {self.loss_fc_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{self.model_name}-{self.loss_fc_name}-metrics_curve.png", dpi=300)
        plt.close()
    
    
if __name__ == '__main__':
    model = EncDec()
    print(model)
    
    x = torch.randn(1, 3, 256, 256)
    y = model(x)