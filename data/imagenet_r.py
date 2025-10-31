import os
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
from .base import ContinualDataset

class SplitImageNetR(ContinualDataset):
    def __init__(self, data_dir: str = "./data/imagenet-r", num_tasks: int = 20, seed: int = 42):
        self.data_dir = data_dir
        self._num_tasks = num_tasks
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self._setup(seed)

    def _setup(self, seed: int):
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Download ImageNet-R to {self.data_dir} from https://github.com/hendrycks/imagenet-r")
        dataset = datasets.ImageFolder(self.data_dir)
        all_classes = list(range(len(dataset.classes)))
        rng = np.random.default_rng(seed)
        rng.shuffle(all_classes)
        self.classes_per_task = np.array_split(all_classes, self._num_tasks)
        self._class_names_per_task = [[f"class_{c}" for c in task] for task in self.classes_per_task]

    def get_task(self, task_id: int):
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        indices = [i for i, (_, label) in enumerate(dataset) if label in self.classes_per_task[task_id]]
        return Subset(dataset, indices)

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def class_names_per_task(self):
        return self._class_names_per_task
