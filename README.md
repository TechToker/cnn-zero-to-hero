# cnn-zero-to-hero

Implementation of iconic CNN backbones from scratch
and compare it on variation of Imagenet100 dataset

Include implementations of:
- Convolutions via numpy
- Depthwise conv
- Residual connections
- etc

Implemented models:

|Model|num param|model size (MB)|acc1|acc5|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|AlexNet|58 691 044|223.89|-|-|
|Vgg11|129 181 540|492.81|78.6|93.2|
|Vgg19|137 630 116|525.05|-|-|
|ResNet18|11 220 132|42.84|81.8|94.6|
|ResNet50|23 712 932|90.66|82.0|94.2|
|MobileNet|3 309 508|12.71|72.6|90.2|
|MobileNetV2|2 351 972|9.10|70.9|89.6|
|MobileNetV3 (L)|1 785 468|6.86|70.9|89.7|
|EfficientNet (B4)|17 727 916|68.11|77.1|92.6|
