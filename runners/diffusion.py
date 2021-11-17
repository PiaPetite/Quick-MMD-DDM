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
from functions.image_utils import generate_sample_sheet, generate_sample_sheet_4, generate_sample_sheet_8
 



def plot_grad_flow(named_parameters, i):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            #print(f'{n}{p.grad.abs().mean().cpu().detach().numpy()}')
            ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
            max_grads.append(p.grad.abs().max().cpu().detach().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout() 
    plt.savefig(f"./gradients/gradient_{i}.png", pad_inches = 9.0)

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0)