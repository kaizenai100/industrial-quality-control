# -*- coding: utf-8 -*-
"""
模型训练脚本
Model training script
"""
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datasets.neu_dataset import NEUDataset
from models.quality_control_cnn import get_model
from config import CHECKPOINTS_DIR, BINARY_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, device
import os

if __name__ == '__main__':
    # 加载数据集
    # Load dataset
    dataset = NEUDataset(BINARY_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    # Initialize model
    model = get_model('quality', num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}')

    # 保存模型
    # Save model
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, 'binary_cnn.pt'))
    print('模型已保存到 checkpoints/binary_cnn.pt')
    # Model saved to checkpoints/binary_cnn.pt