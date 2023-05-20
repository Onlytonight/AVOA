import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#environment
# NSFNet GEANT2 GBN
# gravity_1 gravity_2 gravity_few uniform
ENV_TYPE = 'NSFNet'
TRAFFIC_PROFILE = 'gravity_2'
INIT_SAMPLE = -1

episodes = 200
max_t = 1
ROUTING = 'sp'
ACTION_TYPE = 'all_link_weight'

WEIGTHS_TO_STATES = False
LINK_TRAFFIC_TO_STATES = False
LINK_UTILIZATION_TO_STATES = True
PROBS_TO_STATES = False
K = 6

# 算法模型
AGENT_TYPE='PPO'

#Agent parameters
LEARNING_RATE = 0.0001
GAMMA = 0.6
BETA = 0.7
EPS = 0.5
TAU = 0.99
MODE = 'TD'
SHARE = False
CRITIC = True
NORMALIZE = False
HIDDEN_CONTINUOUS = [256, 256]

#Training parameters
RAM_NUM_EPISODE = 300 #episode
CHANGE_SAMPLE=False
CHANGE_FRAQUNCY=1
SCALE = 1
MAX_T = 100
BATCH_SIZE=32
N_UPDATE = 10 #数据复用次数
UPDATE_FREQUENCY = 1 #更新频次
STATE_STEP=5