"""
二分类模型准确率测试脚本
Binary classification model accuracy test script
"""
import torch
from torch.utils.data import DataLoader
from datasets.neu_dataset import NEUDataset
from models.quality_control_cnn import get_model
from config import CHECKPOINTS_DIR, BINARY_DIR, BATCH_SIZE, device, TEST_TESTB_DIR
from common.enums import BINARY_CLASSES

if __name__ == '__main__':
    # 加载数据集
    # Load dataset
    dataset = NEUDataset()
    dataset.set_samples(TEST_TESTB_DIR, 0)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    # Load model
    model = get_model('quality', num_classes=2).to(device)
    model.load_state_dict(torch.load(f'{CHECKPOINTS_DIR}/binary_cnn.pt', map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'准确率: {100 * correct / total:.2f}%') 