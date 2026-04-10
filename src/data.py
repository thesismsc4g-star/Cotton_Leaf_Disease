from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoProcessor

from config import CLIP_MODEL_NAME, IMAGE_SIZE

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_paths(dataset_dir: Path) -> Tuple[List[str], List[int], List[str], dict]:
    dataset_dir = Path(dataset_dir)
    class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    image_paths: List[str] = []
    labels: List[int] = []
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    for cls in class_names:
        cls_dir = dataset_dir / cls
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                image_paths.append(str(p))
                labels.append(class_to_idx[cls])
    return image_paths, labels, class_names, class_to_idx


class CottonLeafDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose, list, list]:
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_mean = processor.image_processor.image_mean
    clip_std = processor.image_processor.image_std

    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ]
    )

    return train_transform, eval_transform, clip_mean, clip_std


def load_image_for_model(image: Image.Image, eval_transform) -> torch.Tensor:
    return eval_transform(image.convert("RGB")).unsqueeze(0)
