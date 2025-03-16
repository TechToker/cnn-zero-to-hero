import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
    
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    
# Get's input, shrink channels into single value by AvgPool
# Apply FC on it to get value of importance (attention)
# Multiply all channels by attention value (it will be different for each channel)
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)
    
class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, k, stride, hidden_dim, se=False, nl = 'RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert k in [3, 5]
        
        self.identity = stride == 1 and inp == oup
        
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
            
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pointwise
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nlin_layer(inplace=True),
            # depthwise
            nn.Conv2d(hidden_dim, hidden_dim, k, stride, 1 if k == 3 else 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SELayer(hidden_dim),
            nlin_layer(inplace=True),

            # pointwise-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., dropout=0.0):
        super(MobileNetV3, self).__init__()

        self.cfgs = [
            # k, exp, c,  se,     nl,  s,
            [3, 16,  16,  True,  'RE', 2],
            [3, 72,  24,  False, 'RE', 2],
            [3, 88,  24,  False, 'RE', 1],
            [5, 96,  40,  True,  'HS', 2],
            [5, 240, 40,  True,  'HS', 1],
            [5, 240, 40,  True,  'HS', 1],
            [5, 120, 48,  True,  'HS', 1],
            [5, 144, 48,  True,  'HS', 1],
            [5, 288, 96,  True,  'HS', 2],
            [5, 576, 96,  True,  'HS', 1],
            [5, 576, 96,  True,  'HS', 1],
        ]
        
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 4 if width_mult == 0.1 else 8)
        
        last_channel = 1280
        last_channel = _make_divisible(last_channel * width_mult, 8) if width_mult > 1.0 else last_channel
        
        layers = [conv_3x3_bn(3, input_channel, 2)]
        
        # building inverted residual blocks
        block = MobileBottleneck
        
        # building mobile blocks
        for k, exp, c, se, nl, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_channel = _make_divisible(exp * width_mult, 8)
            layers.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
              
        last_conv = _make_divisible(576 * width_mult, 8)
        layers.append(conv_1x1_bn(input_channel, last_conv))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        layers.append(Hswish(inplace=True))
                
        self.features = nn.Sequential(*layers)
        
        # building last several layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes),
        )
            
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
