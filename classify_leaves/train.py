"""
Main file for training model
"""

# import common libraries
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
import math
import wandb
import numpy as np
from tqdm import tqdm
# import libraries about pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import from other file
from dataset import pre_data
from model import initialize_model
from utils import LabelSmoothCELoss, get_device

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

train_name = 'test_accelerator'
# hyperparameter
default_config = dict(
    batch_size=32,
    num_epoch=5,
    learning_rate=3e-4,             # learning rate of Adam
    weight_decay=0.001,             # weight decay 

    warm_up_epochs=10,
    model_path='./model/'+train_name+'_model.ckpt',
    saveFileName='./result/'+train_name+'_pred.csv',
    num_workers=2,
    model_name='effnetv2',
)

wandb.init(project='leaves_classfier', entity='nekokiku', config=default_config, name=train_name)
config = wandb.config

def train(train_loader, val_loader, num_epoch, learning_rate, weight_decay, model_path):

    # Initialize a model, and put it on the device specified.
    model = initialize_model(num_classes=176, model_name=config['model_name'])
    model = model.to(device)
    model.device = device
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = LabelSmoothCELoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    # learning rate schedule
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / config['warm_up_epochs'] if epoch <= config['warm_up_epochs'] else 0.5 * ( math.cos((epoch - config['warm_up_epochs']) /(num_epoch - config['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    # The number of training epochs.
    n_epochs = num_epoch
    best_acc = 0.0

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train() 
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        accelerator.print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        # Iterate the validation set by batches.
        for batch in tqdm(val_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs)

            logits = accelerator.gather(logits)
            labels = accelerator.gather(labels)
            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels)

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # learning rate decay and print 
        scheduler.step()
        realLearningRate = scheduler.get_last_lr()[0]
        # wandb
        wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc':train_acc, 'valid_acc':valid_acc, 'LearningRate':realLearningRate})

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), model_path)

            # torch.save(model.state_dict(), model_path)
            accelerator.print('saving model with acc {:.3f}'.format(best_acc))

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
    print("training")
    train(train_loader, val_loader, num_epoch, learning_rate, weight_decay, model_path)


if __name__ == '__main__':
    main()