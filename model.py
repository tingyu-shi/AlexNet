#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/6 0:23
# @Author: Tingyu Shi
# @File: model.py
# @Description: model


import torch
import torch.nn as nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self):
        """
        引入搭建model所需的函数
        """
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(1)
        self.conv1 = nn.Conv2d(1, 96, 11, 4, 1)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(96, 256, 5, 1, 2)
        self.pool4 = nn.MaxPool2d(3, 2)
        self.conv5 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv6 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv7 = nn.Conv2d(384, 256, 3, 1, 1)
        self.pool8 = nn.MaxPool2d(3, 2)
        self.fc9 = nn.Linear(6 * 6 * 256, 4096)
        self.fc10 = nn.Linear(4096, 4096)
        self.fc11 = nn.Linear(4096, 10)

    def forward(self, x):
        """
        定义前向传播
        :param x: 输入
        :return: x
        """
        x = self.relu(self.conv1(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool4(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool8(x)
        x = self.flatten(x)
        x = self.relu(self.fc9(x))
        x = self.dropout(x)
        x = self.relu(self.fc10(x))
        x = self.dropout(x)
        x = self.softmax(self.fc11(x))
        return x


if __name__ == "__main__":
    """
    测试model的结构
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))
