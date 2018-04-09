from __future__ import print_function, division
import os
from os import listdir
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils

import warnings
warnings.filterwarnings("ignore")

class ImageDataset(Dataset):

    def __init__(self, data_dir, label_dir, transform=None):
        self.label_dir = label_dir
        self.root_dir = data_dir
        self.transform = transform
        self.set_List = listdir(data_dir)
        print(self.set_List)

    def __len__(self):
        return len(self.set_List)

    def __getitem__(self, idx):
        image = self.set_List[idx].split('.')[0]
        items = {}
        discriptionList = listdir(self.label_dir)
        for j in range(len(discriptionList)):
            file = open(self.label_dir+discriptionList[j])
            fileData = file.readlines()
            for i in range(len(fileData)):
                fileName = fileData[i].strip()
                if fileName == image:
                    items = {'image':self.set_List[idx],'discription': discriptionList[j]}
                    print (items)
        return items

set = ImageDataset(data_dir='/home/nas/morozov/oxford/images/', label_dir='/home/nas/morozov/oxford/labels/')
set.__getitem__(74)



