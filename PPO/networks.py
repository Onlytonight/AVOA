import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden=[256, 256]):
        super(Actor, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.mu = nn.Linear(hidden[-1], action_size)
        self.logstd = nn.Linear(hidden[-1], action_size)

    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.softplus(layer(x))
        mu = self.mu(x)
        logstd = self.logstd(x)
        return mu, logstd
    
class Critic(nn.Module):

    def __init__(self, state_size, hidden=[256, 256]):
        super(Critic, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.output = nn.Linear(hidden[-1], 1)
        
    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.softplus(layer(x))
        values = self.output(x)
        return values

class Actor_Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden=[256, 256]):
        super(Actor_Critic, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.mu = nn.Linear(hidden[-1], action_size)
        self.logstd = nn.Linear(hidden[-1], action_size)
        self.output = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.Softplus(layer(x))
        mu = self.mu(x)
        logstd = self.logstd(x)
        values = self.output(x)
        return mu, logstd, values