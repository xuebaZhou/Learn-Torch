import torchvision
from torch import nn

# 调用未预训练的VGG16模型
# 仅创建模型架构，不加载预训练权重
vgg16_false = torchvision.models.vgg16(weights=None)
print(vgg16_false)
print("=============================================================================")
# 调用预训练的VGG16模型
# 直接利用这个已经训练好的模型进行迁移学习、特征提取或作为其他任务的基础模型
vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
print(vgg16_true)

# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print("===============================================================================")
print(vgg16_true)
print("===============================================================================")
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print("===============================================================================")
print(vgg16_false)

vgg16_false.avgpool = nn.AdaptiveAvgPool2d((3, 3))
print(vgg16_false)
