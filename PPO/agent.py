import torch
import torch.optim as optim
import numpy as np

import PPO.networks as networks

import math

class PPO_Agent:
    
    def __init__(self, state_size, action_size, lr, beta, eps, tau, gamma, device, hidden=[256, 256], share=False, mode='MC', use_critic=False, normalize=False):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.share = share
        self.mode = mode
        self.use_critic = use_critic
        self.normalize = normalize
        
        if self.share:
            self.Actor_Critic = networks.Actor_Critic(self.state_size, self.action_size, hidden).to(self.device)
            self.optimizer = optim.Adam(self.Actor_Critic.parameters(), lr)
        else:
            self.Actor = networks.Actor(state_size, action_size, hidden).to(self.device)
            self.Critic = networks.Critic(state_size, hidden).to(self.device)
            self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr)
            self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr)
    def get_agent_info(self):

        info ='PPO'+'-'+'lr'+str(self.lr)+'-'+'beta'+str(self.beta)+'-'+'eps'+str(self.eps)+'-'+'lambda'+str(self.tau)+'-'+'gamma'+str(self.gamma)
        return info
    def act(self, states):
        with torch.no_grad():
            states = torch.tensor(states).view(-1, self.state_size).to(self.device)
            if self.share:
                mu, logstd, _ = self.Actor_Critic(states)
            else:
                mu, logstd = self.Actor(states)

            actions = torch.distributions.Normal(mu, logstd.exp()).sample()
            # actions = torch.sigmoid(actions)
            actions = actions.cpu().numpy().reshape(-1)
        return actions
    
    def process_data(self, states, actions, rewards, batch_size):
        
        actions.append(np.zeros(self.action_size))
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device).view(-1, self.action_size)

        
        #calculate log probabilities and state values
        N = states.size(0) # N-1 is the length of actions, rewards and dones
        log_probs = torch.zeros((N, self.action_size)).to(self.device)
        old_mu = torch.zeros((N, self.action_size)).to(self.device)
        old_logstd = torch.zeros((N, self.action_size)).to(self.device)
        step = math.ceil(N/batch_size)
        
        for ind in range(step):
            if self.share:
                mu, logstd, _ = self.Actor_Critic(states[ind*batch_size:(ind+1)*batch_size, :])
            else:
                mu, logstd = self.Actor(states[ind*batch_size:(ind+1)*batch_size, :])
            distribution = torch.distributions.normal.Normal(mu, logstd.exp())
            log_probs[ind*batch_size:(ind+1)*batch_size, :] = distribution.log_prob(actions[ind*batch_size:(ind+1)*batch_size, :])
            old_mu[ind*batch_size:(ind+1)*batch_size, :] = mu
            old_logstd[ind*batch_size:(ind+1)*batch_size, :] = logstd 
            
        log_probs = log_probs[:-1, :]# remove the last one, which corresponds to no actions
        old_mu = old_mu[:-1, :]
        old_logstd = old_logstd[:-1, :]
        actions = actions[:-1, :]
        log_probs = log_probs.sum(dim=1, keepdim=True)

        rewards = np.array(rewards) #r_t
        
        return states, actions, old_mu.detach(), old_logstd.detach(), log_probs.detach(), rewards
    
    def learn(self, states, actions, old_mu, old_logstd, log_probs, rewards):
        if self.share:
            new_mu, new_logstd, state_values = self.Actor_Critic(states)
            new_mu = new_mu[:-1, :]
            new_logstd = new_logstd[:-1, :]
        else:
            new_mu, new_logstd = self.Actor(states)
            new_mu = new_mu[:-1, :]
            new_logstd = new_logstd[:-1, :]
            state_values = self.Critic(states)
        new_distribution = torch.distributions.normal.Normal(new_mu, new_logstd.exp())
        new_log_probs = new_distribution.log_prob(actions).sum(dim=1, keepdim=True)
        
        KL = new_logstd - old_logstd + (old_logstd.exp().pow(2) + (new_mu - old_mu).pow(2))/(2*new_logstd.exp().pow(2) + 1e-6) - 0.5
        KL = KL.sum(dim=1, keepdim=True)
        
        L = rewards.shape[0]
        with torch.no_grad():
            G = []
            return_value = 0
            if self.mode == 'MC':
                for i in range(L-1, -1, -1):
                    return_value = rewards[i] + self.gamma * return_value
                    G.append(return_value)
                G = G[::-1]
                G = torch.tensor(G, dtype=torch.float).view(-1, 1).to(self.device)
            else:
                rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
                G = rewards +  self.gamma * state_values[1:, :]
            
        Critic_Loss = 0.5*(state_values[:-1, :] - G).pow(2).mean()
        
        with torch.no_grad():
            if self.use_critic:
                G = G - state_values[:-1, :] # advantage
            for i in range(L-2, -1, -1):
                G[i] += G[i+1]*self.gamma*self.tau # cumulated advantage
            if self.normalize:
                G = (G - G.mean()) / (G.std() + 0.00001)
        
        ratio = (new_log_probs - log_probs).exp()
        Actor_Loss1 = ratio * G
        Actor_Loss2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * G
        Actor_Loss = -torch.min(Actor_Loss1, Actor_Loss2)
        Actor_Loss += self.beta * KL

        Actor_Loss = Actor_Loss.mean()

        if self.share:
            Loss = Actor_Loss + Critic_Loss
            self.optimizer.zero_grad()
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Actor_Critic.parameters(), 1)
            self.optimizer.step()
        else:
            self.critic_optimizer.zero_grad()
            Critic_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), 1)
            self.critic_optimizer.step()
            self.actor_optimizer.zero_grad()
            Actor_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 1)
            self.actor_optimizer.step()