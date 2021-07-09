"""
Main file for predict
"""

# import common libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# import libraries about pytorch
import torch
# import from other file
from dataset import pre_data
from model import initialize_model
import utils

import ttach as tta
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

labels_dataframe = pd.read_csv('leaves_data/train.csv')
# Create list of alphabetically sorted labels.
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
#Map each label string to an integer label.
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}

train_path = 'leaves_data/train.csv'
test_path = 'leaves_data/test.csv'
# we already have the iamges floder in the csv file，so we don't need it here
img_path = 'leaves_data/'

train_name = 'test_accelerator'
config = dict(
    batch_size=45,
    num_epoch=5,
    learning_rate=3e-4,             # learning rate of Adam
    weight_decay=0.001,             # weight decay 

    warm_up_epochs=10,
    model_path='./model/'+train_name+'_model.ckpt',
    saveFileName='./result/'+train_name+'_pred.csv',
    num_workers=2,
    model_name='effnetv2',
)

def predict(model_path, test_loader, saveFileName, iftta):

    ## predict
    model = initialize_model(num_classes=176)

    # create model and load weights from checkpoint
    model = model.to(device)
    # model.load_state_dict(torch.load(model_path))

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(model_path))
    model = unwrapped_model
    test_loader = accelerator.prepare(test_loader)

    if iftta:
        print("Using TTA")
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 180]),
                # tta.Scale(scales=[1, 0.3]), 
            ]
        )
        model = tta.ClassificationTTAWrapper(model, transforms)

    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()
    
    # Initialize a list to store the predictions.
    predictions = []
    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):

        imgs = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
            logits = accelerator.gather(logits)

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    preds = []
    for i in predictions:
        preds.append(num_to_class[i])

    test_data = pd.read_csv('leaves_data/test.csv')
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(saveFileName, index=False)
    print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def main():
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    num_epoch = config['num_epoch']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    model_path = config['model_path']
    saveFileName = config['saveFileName']

    print("loading data")
    train_loader, val_loader, test_loader = pre_data(batch_size, num_workers)
    print("testing")
    predict(model_path, test_loader, saveFileName, True)

if __name__ == '__main__':
    main()