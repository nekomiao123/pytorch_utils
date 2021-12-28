"""
Creates a Pytorch dataset to load the datasets
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


train_path = './leaves_data/train.csv'
test_path = './leaves_data/test.csv'
# we already have the iamges floder in the csv file，so we don't need it here
img_path = './leaves_data/'

labels_dataframe = pd.read_csv(train_path)
# Create list of alphabetically sorted labels.
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
# Map each label string to an integer label.
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}

# my onw dataset
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv file path
            img_path (string): image file path 
        """
        # we need resize our images
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # read the csv file using pandas
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None, we can ingore the head
        # length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # The first column is our image file name
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0]) 
            # The second colimn is the label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image 
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])  
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image
            
        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # we can get the file name
        single_image_name = self.image_arr[index]

        # read our image
        img_as_img = Image.open(self.file_path + single_image_name)

        # transform
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),   # Horizontal random flip
                transforms.RandomVerticalFlip(p=0.5),     # Vertical random flip
                transforms.RandomPerspective(p=0.5),
                transforms.RandomAffine(35),
                transforms.RandomRotation(45),            # random rotation
                transforms.RandomGrayscale(p=0.025),      #概率转换成灰度率，3通道就是R=G=B
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean，标准差
            ])
        else:
            # we don't need transfrom for valid and test
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean，标准差
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # image and label

    def __len__(self):
        return self.real_len

def pre_data(batch_size, num_workers):
    train_dataset = LeavesData(train_path, img_path, mode='train')
    val_dataset = LeavesData(train_path, img_path, mode='valid')
    test_dataset = LeavesData(test_path, img_path, mode='test')

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
        )
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    batch_size = 32
    num_workers = 4
    print("loading data")
    train_loader, val_loader, test_loader = pre_data(batch_size, num_workers)
