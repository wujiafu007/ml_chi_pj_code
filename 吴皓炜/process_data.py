import numpy as np
import os
import cv2
import json
import config as config
from pathlib import Path
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from tqdm import  tqdm
from torch import torch
import random
# 处理所有图片
def process_file():
    for dir in tqdm(listdir(config.label_path)):
        for image in listdir(config.label_path+'/'+dir):
            process_image(dir+'/'+image.split('.')[0])
    return
# 处理单张图片
def process_image(path):
    json_path=config.label_path+path+'.json'
    temp_image_path=config.temp_path+path+'.jpg'
    trgt_image_path=config.trgt_path+path+'.jpg'
    data_image_path=config.data_path+path+'.jpg'
    temp_image=cv2.imread(temp_image_path)
    trgt_image=cv2.imread(trgt_image_path)
    if temp_image is None or trgt_image is None:
        return
    temp_image=cv2.resize(temp_image,config.image_size)
    trgt_image=cv2.resize(trgt_image,config.image_size)
    sub_image=cv2.subtract(temp_image,trgt_image)
    Path(config.data_path+path.split('/')[0]).mkdir(parents=True,exist_ok=True)
    cv2.imwrite(data_image_path,sub_image)
    return
# 统计类别用的
def count_label():
    label_count=[0]*15
    for dir in tqdm(listdir(config.label_path)):
        for image in listdir(config.label_path+'/'+dir):
            json_path = config.label_path + dir+'/'+image.split('.')[0] + '.json'
            f = open(json_path)
            data = json.load(f)
            label_count[data['flaw_type']]+=1
            if data['flaw_type']==10:
                print(json_path)
    print(label_count)
    return
# 把图片划分成trainset和testset
def split_set():
    for dir in tqdm(listdir(config.data_path)):
        for image in listdir(config.data_path+'/'+dir):
            path=dir+'/'+image.split('.')[0]
            data_image_path = config.data_path + path + '.jpg'
            data_image=cv2.imread(data_image_path)
            if random.random()<config.trainset_ratio:
                Path(config.train_data_path + path.split('/')[0]).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(config.train_data_path+path+'.jpg',data_image)
            else:
                Path(config.test_data_path + path.split('/')[0]).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(config.test_data_path+path+'.jpg',data_image)
# 把trainset和testset转化成tensor再存储方便后续训练
def data_to_tensor(data):
    count=0
    if data=='train':
        data_path=config.train_data_path
    elif data=='test':
        data_path=config.test_data_path
    for dir in tqdm(listdir(data_path)):
        for image in listdir(data_path+'/'+dir):
            count+=1
    data_image_tensor=torch.zeros(count,3,224,224)
    data_label_tensor=torch.zeros(count)
    count=0
    for dir in tqdm(listdir(data_path)):
        for image in listdir(data_path+'/'+dir):
            path=dir+'/'+image.split('.')[0]
            data_image_path = data_path + path + '.jpg'
            data_json_path=config.label_path+path+'.json'
            data_image=cv2.imread(data_image_path)
            data_label=json.load(open(data_json_path))['flaw_type']
            # data_tensor=torch.cat((data_tensor,torch.tensor(data_image)),-1)
            data_image_tensor[count]=torch.tensor(data_image).permute(2,0,1)
            data_label_tensor[count]=torch.tensor(data_label)
            count+=1
    torch.save(data_image_tensor,config.tensor_path+data+'_image.pt')
    torch.save(data_label_tensor,config.tensor_path+data+'_label.pt')
    return
def data_augmentation():
    # 大佬在这里做supermix
    # svn
    # cutmix
    return
if __name__ == '__main__':
    #count_label()
    process_file()
    data_augmentation()
    split_set()
    data_to_tensor(data='train')
    data_to_tensor(data='test')