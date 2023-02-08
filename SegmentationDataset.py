import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms


from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict
from PIL import Image


class SegmentationDataset(Dataset):

    def __init__(self
                 , data_root: Path
                 , common_augmentations: List = None
                 , image_only_augmentations: List = None
                 , dataset_mode: str = 'train'
                 ):

        self.image_root = data_root / 'images'
        self.mask_root = data_root / 'labels'

        self.common_augmentations = nn.Identity()
        if common_augmentations is not None:
            self.common_augmentations = transforms.Compose(common_augmentations)


        self.image_only_augmentations = nn.Identity()
        if image_only_augmentations is not None:
            self.image_only_augmentations = transforms.Compose(image_only_augmentations)

        self.image_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_to_tensor = transforms.ToTensor()

        self.dataset_mode = dataset_mode
        self.train_dataset = dataset_mode == 'train'

        self.file_list = self.parse_files()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i) -> Dict:
        return self.load_image(*self.file_list[i])

    def load_image(self, im_path: Path, mask_path: Path):

        with im_path.open('rb') as f:
            image = Image.open(f)
            image.load()
            image = image.convert('RGB')

        seed = torch.randint(1000000000, (1,)).item()
        if self.train_dataset:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.common_augmentations(image)
            image = self.image_only_augmentations(image)
        image = self.image_to_tensor(image)

        mask = None

        if mask_path is not None:
            with mask_path.open('rb') as f:
                mask = Image.open(f)
                mask.load()
                mask = mask.convert('1')
            if self.train_dataset:
                random.seed(seed)
                torch.manual_seed(seed)
                mask = self.common_augmentations(mask)

            mask = self.mask_to_tensor(mask)

        return {'image': image, 'mask': mask}

    def parse_files(self):
        images = list(self.image_root.glob('*.png'))
        labels = [self.mask_root / x.relative_to(self.image_root) for x in images]
        labels = [x if x.exists() else None for x in labels]
        path_list = list(zip(images, labels))

        if self.dataset_mode == 'train':
            path_list = [x for x in path_list if x[1] is not None]
        elif self.dataset_mode == 'test':
            path_list = [x for x in path_list if x[1] is None]
        elif self.dataset_mode == 'full':
            pass  # return full dataset
        else:
            raise ValueError(f'dataset_mode {self.dataset_mode} is unknown')
        return path_list
