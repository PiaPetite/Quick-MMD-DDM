import torch
import torch.utils.checkpoint as checkpoint


def compute_alpha(beta, t):

    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0) #add 0 to the beginning of beta
    a = (