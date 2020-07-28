import gym
import torch
import numpy as np
env = gym.make('Ant-v2')
print(env.action_space.high)
print(env.observation_space.high)
print(torch.normal(0.,0.2,(5,6)))
