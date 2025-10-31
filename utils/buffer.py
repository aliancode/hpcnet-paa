from collections import defaultdict
import random
from PIL import Image

class FixedSizeBuffer:
    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.buffer = defaultdict(list)
        self.total = 0

    def add(self, class_id: str, image: Image.Image):
        if self.total < self.max_size:
            self.buffer[class_id].append(image)
            self.total += 1
        else:
            if random.random() < len(self.buffer[class_id]) / self.total:
                idx = random.randrange(len(self.buffer[class_id]))
                self.buffer[class_id][idx] = image
