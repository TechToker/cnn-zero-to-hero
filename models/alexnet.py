import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes, dropout=0):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), stride=4), # 0
            nn.ReLU(inplace=True), 
            nn.MaxPool2d((3, 3), stride=2),
            
            nn.Conv2d(96, 256, (5, 5), padding=2), # 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2),
            
            nn.Conv2d(256, 384, (3, 3), padding=1), # 6
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, (3, 3), padding=1), # 8
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, (3, 3), padding=1), # 10
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2),
            
            # nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.Dropout(dropout),
            nn.Linear(9216, 4096), # 15
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            nn.Linear(4096, 4096), # 18
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes), # 20
        )

    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)

        return out


