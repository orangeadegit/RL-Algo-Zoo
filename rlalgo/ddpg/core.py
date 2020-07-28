import numpy as np
import scipy.signal
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlalgo.models import Mlp, weights_init
from torch.distributions import Normal
from rlalgo.models import tensor2np
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def action4env(action):
    if isinstance(action, torch.Tensor):
        return tensor2np(action)
    return action


class ddpgActor(nn.Module):
    def __init__(self,latent_module,ac_space,indim=256):
        super().__init__()
        assert isinstance(ac_space,)