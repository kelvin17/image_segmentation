import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedBlock(nn.Module):
    """
    Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DilatedNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.layer1 = DilatedBlock(in_channels, 64, dilation=1)
        self.layer2 = DilatedBlock(64, 128, dilation=2)
        self.layer3 = DilatedBlock(128, 256, dilation=4)
        self.layer4 = DilatedBlock(256, 512, dilation=8)
        
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        # upsampling to match the output size
        x = F.interpolate(x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return x

if __name__ == "__main__":
    # 测试网络
    model = DilatedNet(in_channels=3, num_classes=21)
    print(model)

    input_tensor = torch.randn(1, 3, 256, 256)  # batch_size=1, RGB, 256x256
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 应该是 (1, 21, 256, 256)