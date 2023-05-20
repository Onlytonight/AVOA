import numpy as np


class QLearningAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 action,
                 learning_rate=0.9,
                 gamma=0.9,
                 e_greed=0.2,
                 ):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))
        self.action = action

    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.action[obs])
        return action

    def predict(self, obs):
        Q_list = np.array([self.Q[obs][i] for i in self.action[obs]])
        minQ = np.min(Q_list)
        action_list = []
        # print(minQ)
        for i in self.action[obs]:
            # print(self.Q[obs][i])
            if self.Q[obs][i] == minQ:
                action_list.append(i)
        action = np.random.choice(action_list)
        # print(obs,action)
        return action


    def learn(self, obs, action, reward, next_obs, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            Q_list = np.array([self.Q[next_obs][i] for i in self.action[next_obs]])
            minQ = np.min(Q_list)
            target_Q = reward + minQ
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)