"""
NEU表面缺陷数据集加载与预处理
NEU surface defect dataset loading and preprocessing
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_SIZE
from common.enums import BINARY_CLASSES


class NEUDataset(Dataset):
    def __init__(self, root_dir=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # 加载所有图片路径和标签
        # Load all image paths and labels
        if root_dir is not None:
            self.load_samples(root_dir, BINARY_CLASSES)

    def __len__(self):
        return len(self.samples)

    def load_samples(self, root_dir, class_enum):
        # 加载所有图片路径和标签
        # Load all image paths and labels
        if root_dir is not None:
            for label, cls in enumerate(class_enum):
                cls_dir = os.path.join(root_dir, cls)
                if not os.path.exists(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.endswith('.jpg') or fname.endswith('.png'):
                        self.samples.append((os.path.join(cls_dir, fname), label))

    def set_samples(self, cls_dir, label):
        # 设置指定类别的样本
        # Set samples for the specified class
        if os.path.exists(cls_dir):
            for fname in os.listdir(cls_dir):
                if fname.endswith('.jpg') or fname.endswith('.png'):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
        # Convert to grayscale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)  # 转为3通道
        # Convert to 3 channels
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label 