import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Network(nn.Module):

    def __init__(self, state_size, action_size, action_len, hidden=[64, 64]):
        super(Q_Network, self).__init__()
        self.action_size = action_size
        self.action_len = action_len
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size*self.action_len)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(-1, self.action_len, self.action_size)
        return x
