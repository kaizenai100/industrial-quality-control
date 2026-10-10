"""
多分类模型测试脚本，输出整体准确率、每类准确率，并绘制图表
Multi-class model test script, outputs overall accuracy, per-class accuracy, and draws charts
"""
import time

import torch
from torch.utils.data import DataLoader
from datasets.neu_dataset import NEUDataset
from models.quality_control_cnn import get_model
from config import CHECKPOINTS_DIR, MULTI_DIR, TEST_TESTB_DIR, BATCH_SIZE, device, TEST_RESULT_IMAGE_DIR
from common.enums import MULTI_CLASSES
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import os

if __name__ == '__main__':
    # 加载数据集
    # Load dataset
    dataset = NEUDataset()
    dataset.load_samples(MULTI_DIR, MULTI_CLASSES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    # Load model
    model = get_model('quality', num_classes=12).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, 'multi_cnn.pt'), map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 总体准确率
    # Overall accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f'整体准确率: {acc * 100:.2f}%')
    # Overall accuracy

    # 每类准确率
    # Per-class accuracy
    per_class_acc = []
    for i in range(len(MULTI_CLASSES)):
        idx = (all_labels == i)
        if np.sum(idx) == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(np.mean(all_preds[idx] == i))
        print(f'{MULTI_CLASSES[i]} 准确率: {per_class_acc[-1] * 100:.2f}%')
        # Per-class accuracy

    current_stamp = time.time()
    time_struct = time.localtime(current_stamp)
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time_struct)
    # 绘制每类样本总数量柱状图
    # Draw bar chart of sample count per class
    class_counts = [np.sum(all_labels == i) for i in range(len(MULTI_CLASSES))]
    plt.figure(figsize=(10, 6))
    plt.bar(MULTI_CLASSES, class_counts)
    plt.ylabel('count')
    plt.xlabel('class')
    plt.title('per class count')
    plt.xticks(rotation=30)
    plt.tight_layout()
    count_png_name = f'per_class_count_{timestamp_str}.png'
    count_png_path = os.path.join(TEST_RESULT_IMAGE_DIR, count_png_name)
    plt.savefig(count_png_path)
    plt.show()

    # 绘制每类准确率柱状图
    # Draw bar chart of per-class accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(MULTI_CLASSES, per_class_acc)
    plt.ylabel('accuracy')
    plt.xlabel('class')
    plt.title('per class accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    accuracy_png_name = f'per_class_accuracy_{timestamp_str}.png'
    accuracy_png_path = os.path.join(TEST_RESULT_IMAGE_DIR, accuracy_png_name)
    plt.savefig(accuracy_png_path)
    plt.show()

    # 混淆矩阵
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=MULTI_CLASSES, yticklabels=MULTI_CLASSES)
    plt.xlabel('prediction')
    plt.ylabel('real')
    plt.title('Confusion matrix')
    plt.tight_layout()
    matrix_png_name = f'confusion_matrix_{timestamp_str}.png'
    matrix_png_path = os.path.join(TEST_RESULT_IMAGE_DIR, matrix_png_name)
    plt.savefig(matrix_png_path)
    plt.show()