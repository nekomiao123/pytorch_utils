"""
Main file for training model
"""

# import common libraries
import os
import math
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
# ignore the warnings
import warnings
warnings.filterwarnings('ignore')
# import config and logging
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

# for offline running
# os.environ['WANDB_MODE'] = 'dryrun'

def wandb_init(cfg: DictConfig):
    wandb.init(
        project='leaves_classfier', 
        entity='nekokiku',
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    # safe the final config for reproducing
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))

def train(train_loader, val_loader, cfg):
    model_path = os.path.join(os.getcwd(), cfg.exp_name + '_model.pt')
    print(model_path)
    save_file_name = os.path.join(os.getcwd(), cfg.exp_name + '_pred.csv')
    print(save_file_name)

    num_epoch, learning_rate, weight_decay = cfg.num_epochs, cfg.learning_rate, cfg.weight_decay
    warm_up_epochs = cfg.warm_up_epochs

    # Initialize a model, and put it on the device specified.
    model = initialize_model(num_classes=176, model_name=cfg.model_name)
    model = model.to(device)
    model.device = device
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = LabelSmoothCELoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    # learning rate schedule
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( math.cos((epoch - warm_up_epochs) /(num_epoch - warm_up_epochs) * math.pi) + 1)
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
        wandb.log(step=epoch + 1, 
        data={'epoch': epoch + 1, 'train/train_loss': train_loss, 
            'val/val_loss': valid_loss, 'train/train_acc':train_acc, 
            'val/valid_acc':valid_acc, 'LearningRate':realLearningRate})

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), model_path)

            # torch.save(model.state_dict(), model_path)
            accelerator.print('saving model with acc {:.3f}'.format(best_acc))

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg):
    # pretty用来是print出的文字带颜色. 来自于rich这个库
    pretty.install()
    # 把OmniConf的cfg转成yaml，print出来
    print(OmegaConf.to_yaml(cfg))

    wandb_init(cfg)

    print("loading data")
    train_loader, val_loader, test_loader = pre_data(cfg.batch_size, cfg.num_workers)
    print("training")
    train(train_loader, val_loader, cfg)

    # do this after training
    wandb.finish()

if __name__ == '__main__':
    main()