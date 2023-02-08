import torch
import torch.nn as nn
import torchvision.models.segmentation

from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3_ResNet50_Weights


class SegmentationModel(nn.Module):
    def __init__(self):
        """
        Deeplabv3 model with custom head for mask prediction
        """
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier = DeepLabHead(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        y = self.model(x)
        y['out'] = self.sigmoid(y['out'])
        return y



