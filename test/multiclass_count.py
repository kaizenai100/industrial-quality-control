"""
多分类模型测试脚本，输出整体准确率、每类准确率，并绘制图表
Multi-class model test script, outputs overall accuracy, per-class accuracy, and draws charts
"""
import time

import cv2
import torch
from torch.utils.data import DataLoader
from datasets.neu_dataset import NEUDataset
from models.quality_control_cnn import get_model
from torchvision import transforms
from config import CHECKPOINTS_DIR, MULTI_DIRTY_SPOTS_DIR, TEST_TESTB_DIR, BATCH_SIZE, device, IMG_SIZE, TEST_DIR, \
    TEST_RESULT_EXCEL_DIR, TEST_RESULT_IMAGE_DIR, RUN_DIR
from common.enums import MULTI_CLASSES
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
# pip install ultralytics
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载数据集
    # Load dataset
    dataset = NEUDataset()
    dataset.set_samples(MULTI_DIRTY_SPOTS_DIR, 0)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    images = []
    img_paths = []
    defectives = []
    for fname in os.listdir(MULTI_DIRTY_SPOTS_DIR):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(MULTI_DIRTY_SPOTS_DIR, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: {fname} 读取失败，已跳过。")
                # Warning: Failed to read {fname}, skipped.
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Convert to grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
            # Convert to 3 channels
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            images.append(img_tensor)
            img_paths.append(img_path)

    # 加载模型
    # Load model
    model = get_model('quality', num_classes=12).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, 'multi_cnn.pt'), map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for index, inputs in enumerate(images):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

            results = yolo_model(img_paths[index])  # 推理
            for result in results:
                result.show()
            # results.show()  # 显示检测结果

    all_preds = np.array(all_preds)

    current_stamp = time.time()
    time_struct = time.localtime(current_stamp)
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time_struct)
    # 绘制每类样本总数量柱状图
    # Draw bar chart of sample count per class
    class_counts = [np.sum(all_preds == i) for i in range(len(MULTI_CLASSES))]
    plt.figure(figsize=(10, 6))
    plt.bar(MULTI_CLASSES, class_counts)
    plt.ylabel('count')
    plt.xlabel('class')
    plt.title('per class count')
    plt.xticks(rotation=30)
    plt.tight_layout()
    png_name = f'per_class_count_{timestamp_str}.png'
    png_path = os.path.join(TEST_RESULT_IMAGE_DIR, png_name)
    plt.savefig(png_path)
    plt.show()

    # 导出Excel
    # Export to Excel
    df = pd.DataFrame({
        '序号': list(range(1, len(img_paths)+1)),
        '图片路径': img_paths,
        '图片预览': ['']*len(img_paths),
        '分类结果': [MULTI_CLASSES[p] for p in all_preds]
    })
    excel_name = f'results_{time.time()}.xlsx'
    excel_path = os.path.join(TEST_RESULT_EXCEL_DIR, excel_name)
    df.to_excel(excel_path, index=False)

    # 用openpyxl插入图片
    # Insert images using openpyxl
    wb = load_workbook(excel_path)
    ws = wb.active
    for idx, img_path in enumerate(img_paths):
        img = XLImage(img_path)
        img.width = 64
        img.height = 64
        cell = f'C{idx+2}'  # C列，首行为表头
        # Column C, first row is header
        ws.add_image(img, cell)
    wb.save(excel_path)
    print(f'已导出Excel: {excel_path}')