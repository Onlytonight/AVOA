import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np

from DQN.networks import *


class Agent:

    def __init__(self, state_size, action_size, bs, lr, tau, gamma, device, action_len):
        """
        When dealing with visual inputs, state_size should work as num_of_frame
        """
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.action_len = action_len

        self.Q_local = Q_Network(self.state_size, self.action_size, action_len).to(device)
        self.Q_target = Q_Network(self.state_size, self.action_size, action_len).to(device)
        # 使俩个初始值相等
        self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        # 取队列，（s，r，a，ns，d）从后面入队，可以遗忘maxlen以前的，使训练更加平缓
        self.memory = deque(maxlen=100000)

    def act(self, state, eps=0):
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            # 不用生成buffer,速度快
            with torch.no_grad():
                # 计算action的q值
                action_values = self.Q_local(state)
            # 得到最大值的action
            action_v = action_values.cpu()
            return action_v.max(axis=-1).indices[0].data.numpy()
        else:
            # 随机选取动作
            # return random.choice(np.arange(self.action_size))
            # return [random.random() for i in range(42)]
            actions = []
            # 随机从0~10选取42个动作，作为连续动作
            for x in range(self.action_len):
                actions.append(random.choice(np.arange(self.action_size)))
            return actions

    def learn(self):
        experiences = random.sample(self.memory, self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device).reshape(self.bs, self.action_len, 1)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device).reshape(self.bs, self.action_len, 1)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)

        # 计算后取出
        Q_values = self.Q_local(states)
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)

        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            # 取得Q表格一行的最大值，并保持dim
            Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
            Q_targets = rewards + self.gamma * Q_targets

        loss = (Q_values - Q_targets).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            # 线性计算
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
