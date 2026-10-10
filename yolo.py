# -*- coding: utf-8 -*-
"""
单张图片缺陷预测脚本
"""
import argparse

import torch
from ultralytics import YOLO

from models.quality_control_cnn import get_model
from config import CHECKPOINTS_DIR, IMG_SIZE, device
from common.enums import MULTI_CLASSES
import cv2
import numpy as np
from torchvision import transforms
import os

from utils.argument_parser import ArgumentParser
from utils.config_loader import ConfigLoader

if __name__ == '__main__':
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    img_path = args.img_path if args.img_path else config['cnn']['img_path']
    yoloName = args.yolo_name if args.cnn else config['yolo']['yolo_name']

    # 加载预训练模型
    yolo_model = YOLO(os.path.join(CHECKPOINTS_DIR, yoloName))
    # 推理
    results = yolo_model(img_path)
    for result in results:
        result.show()