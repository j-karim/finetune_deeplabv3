import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torchvision.transforms as transforms

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Union, List
from PIL import Image

from Metrics import IOU
from Model import SegmentationModel
from SegmentationDataset import SegmentationDataset


def get_last_modified_model_time_stamp(checkpoint_path: Path):
    """
    :param checkpoint_path: path to the checkpoint folder
    :return the folder name of the most recently trained model
    """
    timestamps = [x for x in checkpoint_path.glob('*') if x.is_dir()]
    timestamps = sorted(timestamps, key=lambda x: x.stat().st_mtime)
    timestamp = timestamps[-1].stem
    return timestamp


def get_last_modified_epoch(checkpoint_path: Path, timestamp: str):
    """

    :param checkpoint_path: path to the checkpoint folder
    :param timestamp: timestamp of the model
    :return: most recently created checkpoint
    """
    epochs = (checkpoint_path / timestamp).glob('*')
    epochs = sorted(epochs, key=lambda x: x.stat().st_mtime)
    epoch = epochs[-1].stem
    return epoch


@dataclass
class PredictionResult:
    path: Path
    image_pil: Image.Image
    image_tensor: torch.Tensor
    ground_truth_mask_pil: Image.Image
    ground_truth_mask_tensor: torch.Tensor
    predicted_mask: torch.Tensor


def evaluate(timestamp: str, epoch: Union[int, str]):
    """
    :param timestamp: timestamp of the model checkpoint
    :param epoch: epoch of the model checkpoint
    :return: list of prediction results (including images, predicted masks and ground truth masks)
    """
    data_path = Path('./data')
    checkpoint_path = Path('./checkpoints')


    model_checkpoint = checkpoint_path / timestamp / str(epoch)

    # load checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_checkpoint, map_location=device)
    model = SegmentationModel()
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    # evaluate on full dataset
    dataset = SegmentationDataset(data_path, dataset_mode='full')
    results = []
    to_image = transforms.ToPILImage()
    for i in tqdm(range(len(dataset))):
        input_dict = dataset[i]
        im = torch.unsqueeze(input_dict['image'], 0)
        gt_mask = input_dict['mask']
        gt_mask_pil = None
        if gt_mask is not None:
            gt_mask_pil = to_image(gt_mask)

        path = dataset.file_list[i][0]
        predicted_mask = model.forward(im)['out']

        im_tensor = im.squeeze(0)
        im_pil = to_image(im_tensor)

        results.append(PredictionResult(path, im_pil, im_tensor, gt_mask_pil, gt_mask, predicted_mask.squeeze(0)))

    return results



def calculate_iou_distribution(results: List[PredictionResult]):
    """
    :param results: list of prediction results
    :return: distribution of intersection over union with corresponding threshold
    """
    iou_metric = IOU()
    gt_masks = torch.concat([x.ground_truth_mask_tensor.unsqueeze(0) for x in results if x.ground_truth_mask_tensor is not None])
    predicted_masks = torch.concat([x.predicted_mask.unsqueeze(0) for x in results if x.ground_truth_mask_tensor is not None])

    thresholds, ious = iou_metric(predicted_masks, gt_masks)
    return thresholds, ious


def dump_latest_results():
    """
    dump predicted masks for most recently trained model
    """
    checkpoint_path = Path('./checkpoints')
    result_path = Path('./results')
    timestamp = get_last_modified_model_time_stamp(checkpoint_path)
    epoch = get_last_modified_epoch(checkpoint_path, timestamp)
    results = evaluate(timestamp, epoch)

    thresholds, ious = calculate_iou_distribution(results)
    best_threshold = thresholds[np.argmax(ious)]
    print(f'Best threshold: {best_threshold}')

    to_image = transforms.ToPILImage()
    out_folder = result_path / timestamp / epoch
    os.makedirs(out_folder, exist_ok=True)

    for r in results:
        mask_image = to_image((r.predicted_mask > best_threshold).float())
        out_path = out_folder / f'{r.path.stem}.png'
        mask_image.save(out_path)



def dump_all_results():
    """
    dump predicted masks for all trained models
    """
    checkpoint_path = Path('./checkpoints')
    result_path = Path('./results')

    timestamps = [x for x in checkpoint_path.glob('*') if x.is_dir()]
    for timestamp in timestamps:
        timestamp = timestamp.stem
        try:
            epoch = get_last_modified_epoch(checkpoint_path, timestamp)
        except IndexError:
            continue

        out_folder = result_path / timestamp / epoch
        if out_folder.exists():
            continue

        results = evaluate(timestamp, epoch)

        thresholds, ious = calculate_iou_distribution(results)
        best_threshold = thresholds[np.argmax(ious)]
        print(f'Best threshold: {best_threshold}')

        to_image = transforms.ToPILImage()
        os.makedirs(out_folder, exist_ok=True)

        for r in results:
            mask_image = to_image((r.predicted_mask > best_threshold).float())
            out_path = out_folder / f'{r.path.stem}.png'
            mask_image.save(out_path)


if __name__ == '__main__':
    dump_all_results()
