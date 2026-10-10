# -*- coding: utf-8 -*-
"""
单张图片缺陷预测脚本
"""
import argparse

import torch
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
    class_type = args.class_type if args.class_type else config['cnn']['class_type']
    img_path = args.img_path if args.img_path else config['cnn']['img_path']
    model_name = args.model_name if args.model_name else config['cnn']['model_name']
    num_classes = args.num_classes if args.num_classes else config['cnn'][class_type]['num_classes']
    cnnName = args.cnn if args.cnn else config['cnn'][class_type]['cnn_name']
    # 读取图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=2)
    img = np.repeat(img, 3, axis=2)
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # 加载模型
    # model_name = 'quality'
    model = get_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, cnnName), map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        print(f'预测结果: {MULTI_CLASSES[pred.item()]}')