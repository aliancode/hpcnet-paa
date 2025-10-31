import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
from .base import ContinualDataset

class SplitCIFAR100(ContinualDataset):
    def __init__(self, data_dir: str = "./data", num_tasks: int = 10, seed: int = 42):
        self.data_dir = data_dir
        self._num_tasks = num_tasks
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        self._setup(seed)

    def _setup(self, seed: int):
        rng = np.random.default_rng(seed)
        all_classes = np.arange(100)
        rng.shuffle(all_classes)
        self.classes_per_task = np.array_split(all_classes, self._num_tasks)
        self._class_names_per_task = [[f"class_{c}" for c in task] for task in self.classes_per_task]

    def get_task(self, task_id: int):
        dataset = datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=self.transform)
        indices = [i for i, (_, label) in enumerate(dataset) if label in self.classes_per_task[task_id]]
        return Subset(dataset, indices)

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def class_names_per_task(self):
        return self._class_names_per_task
