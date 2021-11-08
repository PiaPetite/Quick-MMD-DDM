import os
import logging
import time
import glob
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
from models.diffusion import Model, Model_gradient_checkpointing
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry, KID, KID_rbf
from functions.clip_features import CLIP_fx
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from functions.denoising import generalized_steps, generalized_steps_diff, generalized_steps_gp
from piq.feature_extractors import InceptionV3
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from fid_utils.precalc_fid import calculate_activation_statistics_for_dataloader

EPS = 1e-20

import torchvision.utils as tvu
from functions.image_utils impor