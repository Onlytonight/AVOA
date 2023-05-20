

import random
import numpy as np
from config.config_run_env import *
from env.environment import Environment
from util.util import save_statistic

ENV_TYPE = 'NSFNET'
TRAFFIC_PROFILE = 'gravity_1'
INIT_SAMPLE = -1
ROUTING = 'sp'
env = Environment(env_type=ENV_TYPE, traffic_profile=TRAFFIC_PROFILE, init_sample=INIT_SAMPLE, routing=ROUTING)

action_size = len(env.G.edges)
action = np.ones([action_size])

rewards_log = []
average_log=[]


for e in range(episodes):
    rewards = []
    state = env.get_state() + 1
    for t in range(max_t):
        next_state, reward = env.step(action)
        env_info, traffic, utilization, num_sample = env.get_env_info()
        save_statistic('OSPF-', env_info, traffic, utilization, num_sample, reward)
        state=next_state
        rewards.append(reward)

    env.reset(change_sample=True)

    rewards_log.append([np.sum(rewards),e])
    average_log.append(np.mean(rewards_log[-100:]))
    print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(e, np.sum(rewards), average_log[-1]))

