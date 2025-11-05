import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_conv_2=False):
        super().__init__()
        if use_conv_2:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d(2,2),
                DoubleConv(in_ch, out_ch)
            )
        
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        
        if bilinear:
            # use upsample + conv for
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch //2, kernel_size=2, stride=2)
            # After up, we will concat with skip -> channels = (in_ch//2 + skip_ch)
            self.conv = DoubleConv(in_ch, out_ch)
            
    def forward(self, x1, x2):
        # x2 - skip connection layer
        x1 = self.up(x1)
        
        # pad x1 if necessary to match size, but there is no necessary
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, base_c=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_c)
        # downsampling
        self.down1 = Down(base_c, base_c*2) # 64,128
        self.down2 = Down(base_c*2, base_c*4) # 128,256
        self.down3 = Down(base_c*4, base_c*8) # 256,512
        
        # bottleneck
        self.bottleneck_conv = Down(base_c*8, base_c*8) # 256,512

        # upsampling
        self.up1 = Up(base_c * 16, base_c * 4) # self.up1 = Up(base_c * 16, base_c * 4)
        self.up2 = Up(base_c * 8, base_c * 2) # self.up2 = Up(base_c * 8, base_c * 2)
        self.up3 = Up(base_c * 4, base_c)
        self.up4 = Up(base_c * 2, base_c)
        
        self.outc = OutConv(base_c, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck_conv(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
        
class UNet2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, base_c=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_c)
        # downsampling
        self.down1 = Down(base_c, base_c*2, use_conv_2=True) # 64,128
        self.down2 = Down(base_c*2, base_c*4, use_conv_2=True) # 128,256
        self.down3 = Down(base_c*4, base_c*8, use_conv_2=True) # 256,512
        
        # bottleneck
        self.bottleneck_conv = Down(base_c*8, base_c*8, use_conv_2=True) # 256,512

        # upsampling
        self.up1 = Up(base_c * 16, base_c * 4, bilinear=False) # self.up1 = Up(base_c * 16, base_c * 4)
        self.up2 = Up(base_c * 8, base_c * 2, bilinear=False) # self.up2 = Up(base_c * 8, base_c * 2)
        self.up3 = Up(base_c * 4, base_c, bilinear=False)
        self.up4 = Up(base_c * 2, base_c, bilinear=False)
        
        self.outc = OutConv(base_c, n_classes)
    
    def forward(self, x):
        # print("input:", x.shape)
        x1 = self.inc(x)
        # print("after inc:", x1.shape)
        x2 = self.down1(x1)
        # print("after down1:", x2.shape)
        x3 = self.down2(x2)
        # print("after down2:", x3.shape)
        x4 = self.down3(x3)
        # print("after down3:", x4.shape)
        x5 = self.bottleneck_conv(x4)
        # print("after bottleneck:", x5.shape)
        
        x = self.up1(x5, x4)
        # print("after up1:", x.shape)
        x = self.up2(x, x3)
        # print("after up2:", x.shape)
        x = self.up3(x, x2)
        # print("after up3:", x.shape)
        x = self.up4(x, x1)
        # print("after up4:", x.shape)
        
        logits = self.outc(x)
        return logits


class LightningUNet2(LightningModule):
    def __init__(self, loss_fn=nn.BCEWithLogitsLoss(), loss_name=None, metrics=None, n_channels=3, n_classes=2, base_c=64, with_mask=False):
        super().__init__()
        self.model = UNet2(n_channels=n_channels, n_classes=n_classes, base_c=base_c)
        self.model_name = "UNet"
        self.with_mask = with_mask
        
        self.metrics = metrics
        self.criterium = loss_fn
        self.loss_fc_name = (loss_name if loss_name is not None else loss_fn.__class__.__name__)
        
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
    
    def compute_metrics(self, y_hat, y_true):
        metric_logs = {}
        for name, func in self.metrics.items():
            val = func(y_hat, y_true)
            if isinstance(val, torch.Tensor):
                val = val.item()
            metric_logs[name] = val
        return metric_logs
    
    def compute_metrics_mask(self, y_hat, y_true, mask):
        metric_logs = {}
        for name, func in self.metrics.items():
            val = func(y_hat, y_true, mask)
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
            metrics = self.compute_metrics_mask(outputs, labels, masks)
        else:
            images, labels = batch
            outputs = self(images)
            metrics = self.compute_metrics(outputs, labels)
        
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
    
    ## plot and save
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
    model = UNet2(n_channels=3, n_classes=2, base_c=32)
    print(model)
    
    x = torch.randn(1, 3, 256, 256)
    y = model(x)