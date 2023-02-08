import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex

from Model import SegmentationModel


class LossCalculator:
    def __init__(self, model: SegmentationModel, dataset: Dataset, loss, device):
        """
        Calculates the loss on a given dataset
        """
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=len(dataset))  # dataset fits in memory
        self.loss = loss
        self.device = device

    def __call__(self):
        loss = 0

        for batch in self.loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            prediction = self.model(images)['out']
            batch_loss = self.loss(prediction, masks)
            loss += batch_loss.cpu().item()

        return loss / len(self.loader)


class IOU(nn.Module):
    def __init__(self):
        """
        Calculates the jaccardi index, which is the intersection over union of predicted vs ground truth masks
        """
        super().__init__()
        self.thresholds = np.arange(0., 1., 0.01)
        self.iou_funcs = [JaccardIndex(task='binary', threshold=threshold) for threshold in self.thresholds]


    def forward(self, predicted_mask: torch.Tensor, gt_mask: torch.Tensor):
        predicted_mask = predicted_mask.squeeze()
        gt_mask = gt_mask.squeeze()

        ious = []
        for iou_f in self.iou_funcs:
            iou = iou_f(predicted_mask, gt_mask)
            ious.append(iou)

        return self.thresholds, np.asarray(ious)



