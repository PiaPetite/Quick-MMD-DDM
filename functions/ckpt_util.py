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
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50ef3c6436db52e/?dl=1",
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
    "lsun_bedroom": "diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "ema_lsun_bedroom": "ema_diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "lsun_cat": "diffusion_lsun_cat_model/model-1761000.ckpt",
    "ema_lsun_cat": "ema_diffusion_lsun_cat_model/model-1761000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
    "ema_lsun_church": "ema_diffusion_lsun_church_model/model-4432000.ckpt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
    "lsun_bedroom": "f70280ac0e08b8e696f42cb8e948ff1c",
    "ema_lsun_bedroom": "1921fa46b66a3665e450e42f36c2720f",
    "lsun_cat": "bbee0e7c3d7abfb6e2539eaf2fb9987b",
    "ema_lsun_cat