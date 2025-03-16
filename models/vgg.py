import torch
import torch.nn as nn

class Vgg(nn.Module):
    def __init__(self, num_layers=11, num_classes=100, dropout=0.5):
        super().__init__()

        # VGG11 aka VGG 'A'
        if num_layers == 11:
            config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] # vgg11
        elif num_layers == 19:
            config = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, 512, "M", ]  # vgg19
        else:
            print('Incorrect num_layers')
            return

        layers = []
        in_channels = 3
        
        for v in config:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                bn = nn.BatchNorm2d(v)
                layers += [conv2d, bn, nn.ReLU(inplace=True)]
                in_channels = v

        layers += [nn.AdaptiveAvgPool2d((7, 7))]
        layers += [nn.Flatten()]
        self.features = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(512 * 7 * 7, 4096)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(4096, 4096)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(4096, num_classes)]

        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)
        return out