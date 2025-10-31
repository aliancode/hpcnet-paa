import os
import warnings
from torch.utils.data import Dataset
from .base import ContinualDataset

class MedStream7k(Dataset):
    def __init__(self, root="./data", split="train"):
        self.root = os.path.join(root, "medstream7k")
        if not os.path.exists(self.root):
            warnings.warn(
                "MedStream-7k not found. Please download ChestX-ray14 and ISIC 2019 "
                "and place in data/medstream7k/. Using CIFAR10 as placeholder."
            )
            from torchvision.datasets import CIFAR10
            self.dataset = CIFAR10(root, train=(split=="train"), download=True)
            self.is_stub = True
        else:
            self.dataset = []  # Real implementation would load from folders

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class MedStream7kContinual(ContinualDataset):
    def __init__(self, num_tasks: int = 7):
        self._num_tasks = num_tasks

    def get_task(self, task_id: int):
        return MedStream7k(split="train")

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def class_names_per_task(self):
        return [[f"class_{i}" for i in range(6)] for _ in range(self._num_tasks)]
