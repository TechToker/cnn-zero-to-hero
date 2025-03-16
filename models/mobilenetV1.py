import torch
import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding=1, stride=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, stride=stride, groups=nin, bias=bias)
        self.bn1 = nn.BatchNorm2d(nin)
        
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(nout)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out
    

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        
        self.convolution = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.dws = nn.Sequential(
            depthwise_separable_conv(32, 64, 3, stride=1),
            depthwise_separable_conv(64, 128, 3, stride=2),        
            depthwise_separable_conv(128, 128, 3, stride=1),
            depthwise_separable_conv(128, 256, 3, stride=2),
            depthwise_separable_conv(256, 256, 3, stride=1),
            depthwise_separable_conv(256, 512, 3, stride=2),

            depthwise_separable_conv(512, 512, 3, stride=1),
            depthwise_separable_conv(512, 512, 3, stride=1),
            depthwise_separable_conv(512, 512, 3, stride=1),
            depthwise_separable_conv(512, 512, 3, stride=1),
            depthwise_separable_conv(512, 512, 3, stride=1),

            depthwise_separable_conv(512, 1024, 3, stride=2),
            depthwise_separable_conv(1024, 1024, 3, stride=1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        out = self.convolution(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dws(out)
        
        out = self.avgpool(out)        
        out = self.flatten(out)
        out = self.fc(out)
                
        return out