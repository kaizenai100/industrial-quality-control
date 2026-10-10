from ultralytics import YOLO

from config import CHECKPOINTS_DIR

if __name__ == '__main__':
    model = YOLO(f'{CHECKPOINTS_DIR}/yolov8n.pt')  # 加载预训练模型
    results = model.train(
        data='data.yaml',       # 数据集配置
        epochs=1,               # 训练轮次
        imgsz=640,              # 输入图像尺寸
        batch=16,               # 批大小（根据显存调整）
        optimizer='AdamW',      # 优化器
        lr0=0.001,              # 初始学习率
        name='custom_train'     # 实验名称
    )