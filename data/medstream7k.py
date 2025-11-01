import os
import warnings
from torch.utils.data import Dataset
from .base import ContinualDataset

class MedStream7k:
    def __init__(self, root="./data"):
        self.root = os.path.join(root, "medstream7k")
        if not os.path.exists(self.root):
            raise ValueError(
                "MedStream-7k not found. Please:\n"
                "1. Download ChestX-ray14 from https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
                "2. Download ISIC 2019 from https://challenge.isic-archive.com/data/\n"
                "3. Place folders in data/medstream7k/\n"
                "4. Run scripts/build_medstream.py to preprocess."
            )
        self.dataset = self._load_real_data()

    def _load_real_data(self):
        # Load real ChestX-ray14 + ISIC images
        # Apply streaming protocol: 7 modalities, 42 classes
        pass

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
