# 工业缺陷智能检测与分类系统

本项目致力于工业产品表面缺陷的自动化检测与多分类，支持多种主流深度学习模型结构，适用于工业质检、缺陷定位等场景。

## 特性
- 支持多分类模型（12类：无瑕疵+11种缺陷类型）
- 支持三种主干网络可选：SimpleCNN（自定义CNN）、MobileNetV2、EfficientNetB0
- 支持测试结果可视化（准确率、混淆矩阵、样本分布等）
- 支持检测结果导出为带图片预览的Excel

## 目录结构
```
├── models/                # 各类模型结构定义
│   └── quality_control_cnn.py      # 包含SimpleCNN、MobileNetV2、EfficientNetB0及工厂方法
├── datasets/              # 数据集处理
│   └── neu_dataset.py     # 数据集加载与预处理
├── train/                 # 模型训练脚本
│   ├── binary_train.py    # 二分类模型训练脚本
│   ├── multi_train.py     # 多分类模型训练脚本
├── test/                  # 测试与可视化脚本
│   ├── binary_test.py            # 二分类模型准确率测试
│   ├── multiclass_accuracy.py    # 多分类模型准确率测试（含图表）
│   └── multiclass_count.py       # 多分类统计与Excel导出
├── checkpoints/           # 训练好的模型权重
├── utils/                 # 工具函数
├── config.py              # 配置文件（路径、参数等）
├── common/                # 公共枚举、工具等
│   └── enums.py           # 类别名等
├── data/                  # 数据集目录
│   └── binary/            # 二分类数据集
│   └── multi/             # 多分类数据集
│   └── test/              # 测试数据集
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明
```

## 支持的模型结构
- `SimpleCNN`：自定义轻量级卷积神经网络
- `MobileNetV2Classifier`：高效轻量，适合速度要求高的场景
- `EfficientNetB0Classifier`：高效且精度较高
- 通过`get_model(model_name, num_classes=12, pretrained=True)`工厂方法一键切换

## 缺陷类别
- 0-无瑕疵
- 1-不导电
- 2-凸粉
- 3-擦花
- 4-桔皮
- 5-横条压凹
- 6-涂层开裂
- 7-漏底
- 8-碰伤
- 9-脏点
- 10-起坑
- 11-其他

## 数据集格式
- **分类任务**：图片按类别分文件夹
- **检测任务**：图片按类别分文件夹

## 主要依赖
- torch, torchvision
- numpy, pandas, matplotlib, seaborn
- opencv-python
- scikit-learn
- openpyxl

## 运行环境建议
- Python 3.8+
- 推荐使用GPU加速

## 安装依赖包：
```
pip install -r requirements.txt
```
generated_yolo
## 数据准备
- 下载数据集并解压到`data/`目录，结构如下：
  ```
  data/
    binary/
    multi/
    test/
  ```
  `defective`为有缺陷样本，`normal`为无缺陷样本。

## 训练与测试
### 训练（以多分类为例）
```sh
python train/multi_train.py
```

### 测试与可视化
```sh
python test/multiclass_accuracy.py
```
- 输出整体准确率、每类准确率、样本分布等图表

### 检测结果导出Excel
```sh
python test/multiclass_count.py
```
- 自动生成`results.xlsx`，含图片预览和分类结果

### 单图片缺陷检查
```sh
python cnn.py
# 参数
## --class_type: binary, multi
## --img_path: picture path
## --model_name: quality, mobilenetv2, efficientnetb0
```
- 生成预测结果

### YOLO模型单图片检查缺陷
```sh
python yolo.py
# 参数
## --img_path: picture path
```
- 生成预测结果

## 联系与支持
如有问题或定制需求，请联系项目维护者。 