#! /usr/bin/python

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Metrics import LossCalculator
from SegmentationDataset import SegmentationDataset
from Model import SegmentationModel


BATCH_SIZE = 4
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
TRAIN = True


def train():
    """
    Training script for fine-tuning a model with the rooftop dataset
    """

    # cuda setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Cuda: {torch.cuda.is_available()}')

    # tensorboard
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%dT%H:%M:%S')
    summary_writer = SummaryWriter(f'./logs/{timestamp}')

    # path definition
    data_path = Path('./data')
    checkpoint_path = Path('./checkpoints')
    os.makedirs(checkpoint_path / timestamp)

    # common augmentations (apply to image *and* mask)
    common_augmentations = None
    common_augmentations = [
        transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(256, (0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]

    # augmentations (apply to image only)
    image_only_augmentations = None
    # image_only_augmentations = [
    #     transforms.RandomAdjustSharpness(0.7),
    # ]

    # dataset
    dataset = SegmentationDataset(data_path, common_augmentations, image_only_augmentations, dataset_mode='train')
    train_dataset, val_dataset = random_split(dataset, [20, 4])
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # model
    model = SegmentationModel()
    model = model.to(device)
    model.train()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # loss
    loss_func = nn.BCELoss()
    val_loss_calculator = LossCalculator(model, val_dataset, loss_func, device)

    min_val_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}')
        train_loss = 0
        for i, batch in tqdm(enumerate(data_loader)):
            step = epoch*len(data_loader)+i

            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # add input images to tensorboard
            summary_writer.add_images('Images', images, global_step=step)
            summary_writer.add_images('Masks', masks, global_step=step)
            summary_writer.add_images('Overlay', (masks + images) / 2.0, global_step=step)

            # optimization step
            prediction = model(images)['out']
            loss = loss_func(nn.Flatten()(prediction), nn.Flatten()(masks))
            loss.backward()
            optimizer.step()

            # add predictions to tensorboard
            summary_writer.add_images('Binarized masks', (prediction > 0.5).float(), global_step=step)
            summary_writer.add_images('Predicted masks', prediction, global_step=step)
            train_loss += loss.cpu().item()

        # calculate validation loss and save model depending on validation loss
        val_loss = val_loss_calculator()
        train_loss /= len(data_loader)
        if val_loss < min_val_loss:
            torch.save({
                'model_state_dict': model.state_dict()
            }, checkpoint_path / timestamp / 'best_model')

        # add loss values to tensorboard
        summary_writer.add_scalars('Loss', {'Train loss': train_loss, 'Validation loss': val_loss}, global_step=epoch)
        summary_writer.flush()

        # save epoch checkpoint
        torch.save({
            'model_state_dict': model.state_dict()
        }, checkpoint_path / timestamp / str(epoch).zfill(5))


if __name__ == '__main__':
    train()
