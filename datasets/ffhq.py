from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class FFHQ(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
 