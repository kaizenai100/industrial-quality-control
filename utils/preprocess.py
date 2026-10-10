"""
数据预处理工具
Data preprocessing utilities
"""
import cv2
import numpy as np

def resize_image(img_path, size):
    # 读取图片
    # Read image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    return img

def normalize_image(img):
    # 归一化到0-1
    # Normalize to 0-1
    return img / 255.0

def to_gray(img):
    """转为灰度图"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, ksize=5):
    """高斯模糊去噪"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def canny_edge(img, low=100, high=200):
    """Canny边缘检测"""
    return cv2.Canny(img, low, high)

def threshold(img):
    """自适应阈值分割"""
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2) 