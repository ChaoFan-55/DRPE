from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import ExperimentConfig
from .utils import seed_worker


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


DATASET_SPECS = {
    "cifar10": DatasetSpec("cifar10", 10, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": DatasetSpec("cifar100", 100, 32, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tiny_imagenet": DatasetSpec("tiny_imagenet", 200, 64, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


@dataclass
class FederatedData:
    client_loaders: List[DataLoader]
    client_sizes: List[int]
    val_loader: DataLoader
    test_loader: DataLoader
    client_indices: List[List[int]]


def get_dataset_spec(dataset: str) -> DatasetSpec:
    key = dataset.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASET_SPECS[key]


def dataset_transforms(dataset: str):
    spec = get_dataset_spec(dataset)
    train_ops = []
    if spec.image_size == 32:
        train_ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    else:
        train_ops.extend([transforms.RandomCrop(spec.image_size, padding=4), transforms.RandomHorizontalFlip()])
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)])
    eval_ops = [transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: str | Path, transform=None, class_to_idx: dict[str, int] | None = None):
        self.root = Path(root)
        self.transform = transform
        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Tiny-ImageNet wnids.txt not found under {self.root}")
        wnids = [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.class_to_idx = class_to_idx or {name: index for index, name in enumerate(sorted(wnids))}
        annotations = self.root / "val" / "val_annotations.txt"
        images_dir = self.root / "val" / "images"
        if not annotations.exists():
            raise FileNotFoundError(f"Tiny-ImageNet validation annotations not found: {annotations}")
        self.samples: list[tuple[Path, int]] = []
        for line in annotations.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            self.samples.append((images_dir / filename, self.class_to_idx[wnid]))
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _tiny_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    direct = root / "tiny-imagenet-200"
    if direct.exists():
        return direct
    if root.name == "tiny-imagenet-200" and root.exists():
        return root
    raise FileNotFoundError(
        "Tiny-ImageNet was not found. Place it at ./data/tiny-imagenet-200 "
        "with train/, val/, wnids.txt and words.txt."
    )


def build_dataset_views(cfg: ExperimentConfig):
    train_transform, eval_transform = dataset_transforms(cfg.dataset)
    if cfg.dataset == "cifar10":
        train = datasets.CIFAR10(cfg.data_root, train=True, download=cfg.dataset_download, transform=train_transform)
        eval_train = datasets.CIFAR10(cfg.data_root, train=True, download=cfg.dataset_download, transform=eval_transform)
        test = datasets.CIFAR10(cfg.data_root, train=False, download=cfg.dataset_download, transform=eval_transform)
        labels = np.asarray(train.targets)
    elif cfg.dataset == "cifar100":
        train = datasets.CIFAR100(cfg.data_root, train=True, download=cfg.dataset_download, transform=train_transform)
        eval_train = datasets.CIFAR100(cfg.data_root, train=True, download=cfg.dataset_download, transform=eval_transform)
        test = datasets.CIFAR100(cfg.data_root, train=False, download=cfg.dataset_download, transform=eval_transform)
        labels = np.asarray(train.targets)
    else:
        root = _tiny_root(cfg.data_root)
        train = datasets.ImageFolder(root / "train", transform=train_transform)
        eval_train = datasets.ImageFolder(root / "train", transform=eval_transform)
        test = TinyImageNetValDataset(root, transform=eval_transform, class_to_idx=train.class_to_idx)
        labels = np.asarray(train.targets)
    return train, eval_train, test, labels


def build_attack_dataset(dataset: str, data_root: str, split: str, download: bool = True):
    raw = transforms.ToTensor()
    if dataset == "cifar10":
        return datasets.CIFAR10(data_root, train=split == "train", download=download, transform=raw)
    if dataset == "cifar100":
        return datasets.CIFAR100(data_root, train=split == "train", download=download, transform=raw)
    root = _tiny_root(data_root)
    train = datasets.ImageFolder(root / "train", transform=raw)
    if split == "train":
        return train
    return TinyImageNetValDataset(root, transform=raw, class_to_idx=train.class_to_idx)


def stratified_train_val_split(labels: np.ndarray, validation_ratio: float, seed: int):
    if validation_ratio <= 0:
        all_indices = np.arange(len(labels))
        return all_indices, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        val_count = int(round(len(cls_indices) * validation_ratio))
        val_parts.append(cls_indices[:val_count])
        train_parts.append(cls_indices[val_count:])
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def dirichlet_partition(
    labels: np.ndarray,
    candidate_indices: Sequence[int],
    num_clients: int,
    alpha: float,
    min_samples_per_client: int,
    seed: int,
    max_attempts: int = 200,
) -> List[List[int]]:
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in np.unique(labels[candidate_indices]):
            cls_indices = candidate_indices[labels[candidate_indices] == cls].copy()
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(cls_indices, split_points)):
                client_indices[client_id].extend(split.tolist())
        if min(map(len, client_indices)) >= min_samples_per_client:
            for values in client_indices:
                rng.shuffle(values)
            return client_indices
    raise RuntimeError("Could not create a valid Dirichlet split. Reduce min_samples_per_client or increase alpha.")


def build_federated_data(cfg: ExperimentConfig) -> FederatedData:
    spec = get_dataset_spec(cfg.dataset)
    if cfg.num_classes != spec.num_classes:
        raise ValueError(f"num_classes={cfg.num_classes} does not match {cfg.dataset} ({spec.num_classes}).")
    train_dataset, eval_train_dataset, test_dataset, labels = build_dataset_views(cfg)
    train_indices, val_indices = stratified_train_val_split(labels, cfg.validation_ratio, cfg.partition_seed)
    client_indices = dirichlet_partition(
        labels,
        train_indices,
        cfg.num_clients,
        cfg.dirichlet_alpha,
        cfg.min_samples_per_client,
        cfg.partition_seed + 17,
    )
    client_loaders: List[DataLoader] = []
    for client_id, indices in enumerate(client_indices):
        generator = torch.Generator().manual_seed(cfg.seed + 10_000 + client_id)
        client_loaders.append(DataLoader(
            Subset(train_dataset, indices),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
            generator=generator,
            drop_last=False,
        ))
    common = dict(
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(Subset(eval_train_dataset, val_indices.tolist()), **common)
    test_loader = DataLoader(test_dataset, **common)
    return FederatedData(
        client_loaders=client_loaders,
        client_sizes=[len(indices) for indices in client_indices],
        val_loader=val_loader,
        test_loader=test_loader,
        client_indices=client_indices,
    )
