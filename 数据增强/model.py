import random

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
from PIL import Image
from sklearn.model_selection import KFold
import da
class Pj_Dataset(Dataset):
    def __init__(self, mode='train',da_strategy=0,da_times=1):
        self.image=torch.load(config.tensor_path+mode+'_image.pt')
        self.label=torch.load(config.tensor_path+mode+'_label.pt').type(torch.LongTensor)
        self.da_strategy=da_strategy
    def __len__(self):
        return self.label.shape[0]
    def __getitem__(self, idx):
        if random.random()<0.5:
            return self.image[idx].cuda(), self.label[idx].cuda()
        else:
            return self.data_augmentation(self.image[idx]).cuda(),self.label[idx].cuda()
    def data_augmentation(self,image):
        return da.da_by_strategy(image,self.da_strategy)

class res_fc(nn.Module):
    def __init__(self):
        super(res_fc, self).__init__()
        self.fc = nn.Linear(512, 15)
    def forward(self, x):
        x = F.relu(self.fc(x))
        return x
def k_fold_split(dataset,k_fold):


    return
def train(da_strategy=0,k_fold=5):
    data_set=Pj_Dataset(mode='train')
    kfold=KFold(n_splits=k_fold,shuffle=True)
    accuracy=[]
    macro_f1=[]
    for fold,(train_ids,val_ids) in enumerate(kfold.split(data_set)):
        train_subsampler=torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler=torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader=DataLoader(data_set,batch_size=config.batch_size,sampler=train_subsampler)
        val_loader=DataLoader(data_set,batch_size=1,sampler=val_subsampler)
        model = models.resnet18(pretrained=False)
        model.fc = res_fc()
        model=nn.DataParallel(model)
        model = model.to(torch.device('cuda'))
        criterion = nn.CrossEntropyLoss()
        optimzer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        model.train()
        for epoch in range(config.epoch):
            # loss_sum=0.
            # print(f'epoch:{epoch}')
            for index, (image_tensor, image_label) in enumerate(tqdm(train_loader, desc=f'fold:{fold}epoch:{epoch}')):
                y_label = image_label
                y_pred = model(image_tensor)
                loss = criterion(y_pred, y_label)
                # loss_sum += loss.item()
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
            # losses.append(loss_sum / len() * config.batch_size)
        # val_losses.append(model_eval(model))
        eval_accuracy,eval_macro_f1=model_eval(model,val_loader)
        print(eval_accuracy)
        print(eval_macro_f1)
        accuracy.append(eval_accuracy)
        macro_f1.append(eval_macro_f1)
    average_accuracy=sum(accuracy)/len(accuracy)
    average_macro_f1=sum(macro_f1) / len(macro_f1)
    print("average accuracy:{:.4f}".format(average_accuracy))
    print("average macro_f1:{:.4f}".format(average_macro_f1))
    save_res(model,da_strategy,average_accuracy,average_macro_f1)
    # ax = plt.gca()
    # ax.set_ylim([0, 10])
    # plt.plot(losses, color='b', label='train_loss')
    # plt.plot(val_losses, color='r', label='test_loss')
    # plt.legend()
    # plt.show()
    return
def model_eval(model,val_loader):
    model.eval()
    correct_count = 0.
    test_case_count = 0.
    case_count = torch.zeros(15, 15, dtype=int)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.
    f1_pred=[]
    f1_real=[]
    for index, (image_tensor, image_label) in enumerate(val_loader):
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
    macro_f1=f1_score(f1_real,f1_pred,average='macro')
    # print("accurancy:{:.4f}".format(accuracy))
    # print(macro_f1)
    # print(case_count)
    return accuracy,macro_f1
def save_res(model,da_strategy,accuracy,macro_f1):
    if not os.path.isdir(config.res_save_path):
        os.mkdir(config.res_save_path)
    with open(config.res_save_path+'result.txt',"a+") as f:
        f.write(f'da_strategy:{da.da_strategies[da_strategy].__name__}  accuracy:{accuracy}  macro_f1:{macro_f1} \n')
    return
if __name__ == '__main__':
    for strategy in tqdm(range(len(da.da_strategies)),desc='da_strategy:'):
        train(da_strategy=strategy,k_fold=config.k_fold)
