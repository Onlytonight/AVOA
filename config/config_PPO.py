import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#env names
RAM_DISCRETE_ENV_NAME = 'LunarLander-v2'
RAM_CONTINUOUS_ENV_NAME = 'LunarLanderContinuous-v2'
CONSTANT = 90

# 算法模型
AGENT_TYPE='PPO'

#Agent parameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BETA = 0.5
EPS = 0.1
TAU = 0.99
MODE = 'TD'
SHARE = False
CRITIC = True
NORMALIZE = False
HIDDEN_CONTINUOUS = [256, 256]

#Training parameters
RAM_NUM_EPISODE = 2000
VISUAL_NUM_EPISODE = 5000
SCALE = 1
MAX_T = 100
NUM_FRAME = 2
N_UPDATE = 10
UPDATE_FREQUENCY = 1
