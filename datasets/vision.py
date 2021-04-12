import os
import torch
import torch.utils.data as data


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def