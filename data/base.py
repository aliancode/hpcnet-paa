from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class ContinualDataset(ABC):
    @abstractmethod
    def get_task(self, task_id: int) -> Dataset:
        pass

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        pass

    @property
    @abstractmethod
    def class_names_per_task(self):
        pass
