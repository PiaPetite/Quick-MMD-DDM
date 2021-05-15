import torch
import torch.utils.checkpoint as checkpoint


def compute_alpha(beta, t):

    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0) #add 0 to the beginning of beta
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1) #shape (n x 1 x 1 x 1)
    return a

def custom(module):
    def custom_forward(*inputs):
        out =  module(inputs[0], inputs[1])
        return out
    return custom_forward

def generalized_steps_diff(x, seq, model, b, **kwargs):

    """
    Function that samples from the model using the generalized DDPM sampling scheme. It takes as input the initial noise x
    and the sequence of time steps to sample from. The seq is unique for each image in the batch.
    seq has shape (seq_len)
    """
    
    n = x.size(0)
    seq_next = [-1] + list(seq[: