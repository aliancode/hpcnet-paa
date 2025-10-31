from datasets import load_dataset
from torch.utils.data import Subset
from .base import ContinualDataset
import numpy as np

class CORe50(ContinualDataset):
    def __init__(self, num_tasks: int = 10, seed: int = 42):
        self._num_tasks = num_tasks
        self._setup(seed)

    def _setup(self, seed: int):
        rng = np.random.default_rng(seed)
        obj_ids = np.arange(50)
        rng.shuffle(obj_ids)
        self.classes_per_task = np.array_split(obj_ids, self._num_tasks)
        self._class_names_per_task = [[f"class_{c}" for c in task] for task in self.classes_per_task]

    def get_task(self, task_id: int):
        ds = load_dataset("core50", "core50_128", split="train")
        task_objs = set(self.classes_per_task[task_id])
        indices = [i for i, item in enumerate(ds) if item["object_id"] in task_objs]
        return Subset(ds, indices)

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def class_names_per_task(self):
        return self._class_names_per_task
