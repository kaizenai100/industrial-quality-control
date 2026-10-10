"""
简单卷积神经网络模型，用于工业缺陷二分类
Simple convolutional neural network model for binary classification of industrial defects
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 第一层卷积，输入3通道，输出16通道
        # First convolutional layer, input 3 channels, output 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积，输入16通道，输出32通道
        # Second convolutional layer, input 16 channels, output 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 全连接层
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 卷积+激活+池化
        # Convolution + activation + pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        # Flatten
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super(MobileNetV2Classifier, self).__init__()
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super(EfficientNetB0Classifier, self).__init__()
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(model_name, num_classes=12, pretrained=True):
    """
    根据模型名返回对应的模型实例。
    Return the corresponding model instance according to the model name.
    model_name: 'quality', 'mobilenetv2', 'efficientnetb0'
    """
    if model_name == 'quality':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'mobilenetv2':
        return MobileNetV2Classifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnetb0':
        return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f'Unknown model name: {model_name}')