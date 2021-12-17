import numpy as np
import os
import cv2
import json
import config as config
import matplotlib.pyplot as plt
from tqdm import  tqdm
import torch
import torch.nn as nn
from torch import torch
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
class Pj_Dataset(Dataset):
    def __init__(self, mode='train'):
        self.image=torch.load(config.tensor_path+mode+'_image.pt')
        self.label=torch.load(config.tensor_path+mode+'_label.pt').type(torch.LongTensor)
    def __len__(self):
        return self.label.shape[0]
    def __getitem__(self, idx):
        return self.image[idx].cuda(), self.label[idx].cuda()

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )
        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)
class res_fc(nn.Module):
    def __init__(self):
        super(res_fc, self).__init__()
        self.fc = nn.Linear(512, 15)
    def forward(self, x):
        x = F.relu(self.fc(x))
        return x
def train():
    model=models.vgg34(pretrained=False)
    model.fc=res_fc()
    model=model.to(torch.device('cuda'))
    criterion=nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_set=Pj_Dataset(mode='train')
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    losses = []
    val_losses = []
    model.train()
    for epoch in range(config.epoch):
        loss_sum=0.
        print(f'epoch:{epoch}')
        for index, (image_tensor, image_label) in enumerate(tqdm(train_loader, desc='epoch:{}'.format(epoch))):
            y_label = image_label
            y_pred = model(image_tensor)
            loss = criterion(y_pred, y_label)
            loss_sum += loss.item()
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        losses.append(loss_sum / len(train_set) * config.batch_size)
        val_losses.append(test_model(model))
    ax = plt.gca()
    # ax.set_ylim([0, 10])
    plt.plot(losses, color='b', label='train_loss')
    plt.plot(val_losses, color='r', label='test_loss')
    plt.legend()
    plt.show()
    return
def test_model(model):
    model.eval()
    correct_count = 0.
    test_case_count = 0.
    case_count = torch.zeros(15, 15, dtype=int)
    test_set = Pj_Dataset(mode='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.
    f1_pred=[]
    f1_real=[]
    for index, (image_tensor, image_label) in enumerate(tqdm(test_loader, desc='test:')):
        test_case_count += 1
        y_pred = model(image_tensor)
        loss = criterion(y_pred, image_label)
        loss_sum += loss.item()
        predict_label = torch.max(y_pred.data, 1)[1]
        if predict_label == image_label.item():
            correct_count += 1
        case_count[image_label.item(), predict_label] += 1
        f1_pred.append(predict_label.item())
        f1_real.append(image_label.item())
    accuracy = correct_count / test_case_count
    print("accurancy:{:.4f}".format(accuracy))
    print(f1_score(f1_real,f1_pred,average='macro'))
    # print(case_count)
    return loss_sum / test_case_count
if __name__ == '__main__':
    train()