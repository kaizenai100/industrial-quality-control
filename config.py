"""
配置文件，包含路径、超参数等
"""
import os

# 模型相关路径
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
# 数据集相关路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RUN_DIR = os.path.join(os.path.dirname(__file__), 'runs')
## 二分类相关
BINARY_DIR = os.path.join(DATA_DIR, 'binary')
BINARY_DEFECTIVE_DIR = os.path.join(BINARY_DIR, 'defective')
BINARY_NORMAL_DIR = os.path.join(BINARY_DIR, 'normal')
## 多分类相关
MULTI_DIR = os.path.join(DATA_DIR, 'multi')
MULTI_NORMAL_DIR = os.path.join(MULTI_DIR, 'normal')
MULTI_NON_CONDUCTIVE_DIR = os.path.join(MULTI_DIR, 'nonconductive')
MULTI_CONVEX_POWDER_DIR = os.path.join(MULTI_DIR, 'convexpowder')
MULTI_RUB_THE_FLOWERS_DIR = os.path.join(MULTI_DIR, 'rubtheflowers')
MULTI_ORANGE_PEEL_DIR = os.path.join(MULTI_DIR, 'orangepeel')
MULTI_CONCAVE_DIR = os.path.join(MULTI_DIR, 'concave')
MULTI_CRACKING_COATING_DIR = os.path.join(MULTI_DIR, 'crackingcoating')
MULTI_LEAKY_BOTTOM_DIR = os.path.join(MULTI_DIR, 'leakybottom')
MULTI_BUMPS_DIR = os.path.join(MULTI_DIR, 'bumps')
MULTI_DIRTY_SPOTS_DIR = os.path.join(MULTI_DIR, 'dirtyspots')
MULTI_PITTING_DIR = os.path.join(MULTI_DIR, 'pitting')
MULTI_OTHER_DIR = os.path.join(MULTI_DIR, 'other')
## 测试相关
TEST_DIR = os.path.join(DATA_DIR, 'test')
TEST_TESTA_DIR = os.path.join(TEST_DIR, 'testa')
TEST_TESTB_DIR = os.path.join(TEST_DIR, 'testb')
TEST_TESTC_DIR = os.path.join(TEST_DIR, 'testc')
TEST_RESULT_DIR = os.path.join(TEST_DIR, 'results')
TEST_RESULT_EXCEL_DIR = os.path.join(TEST_RESULT_DIR, 'excels')
TEST_RESULT_IMAGE_DIR = os.path.join(TEST_RESULT_DIR, 'images')

## yolo
YOLO_DIR = os.path.join(DATA_DIR, 'yolo')
YOLO_IMAGE_DIR = os.path.join(YOLO_DIR, 'images')
YOLO_IMAGE_TRAIN_DIR = os.path.join(YOLO_IMAGE_DIR, 'train')
YOLO_LABEL_DIR = os.path.join(YOLO_DIR, 'labels')
YOLO_LABEL_TRAIN_DIR = os.path.join(YOLO_LABEL_DIR, 'train')

# 训练参数
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 128


# 设备
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 