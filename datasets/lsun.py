from .vision import VisionDataset
from PIL import Image
import os
import os.path
import io
from collections.abc import Iterable
import pickle
from torchvision.datasets.utils import verify_str_arg, iterable_to_str


class LSUNClass(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        import lmdb

        super(LSUNClass, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.env = lmdb.o