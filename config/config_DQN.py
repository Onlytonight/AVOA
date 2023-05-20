import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#env names
MY_ENV = 'PPO_ENV'
RAM_ENV_NAME = 'LunarLander-v2'
VISUAL_ENV_NAME = 'Pong-v0'
CONSTANT = 90

#Agent parameters
STATE_SIZE = 42
ACTION_SIZE = 10
# 连续动作个数即链路数
ACTION_LEN = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99

#Training parameters
RAM_NUM_EPISODE = 1000
EPS_INIT = 1
EPS_DECAY = 0.995
EPS_MIN = 0.05
MAX_T = 100
# 实际可能做了2-6个frame
NUM_FRAME = 2
