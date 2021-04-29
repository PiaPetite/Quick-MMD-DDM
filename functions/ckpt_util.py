import os, hashlib
import requests
from tqdm import tqdm

URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
    "lsun_bedroom": "https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1",
    "ema_lsun_bedroom": "https://heibox.uni-heidelberg.de/f/b95206528f384185889b/?dl=1",
    "lsun_cat": "https://heibox.uni-heidelberg.de/f/fac870bd988348eab88e/?dl=1",
    "ema_lsun_cat": "https://heibox.uni-heidelberg.de/f/0701aac3aa69457bbe34/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50e