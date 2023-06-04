import numpy as np
import networkx as nx
from env.environment import Environment
from Q_learning.agent import QLearningAgent
from util.util import save_statistic
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)


def train(action_size):
    routings = []
    for episode in range(200):
        # print(episode, 'change demand')
        if episode == 0:
            env.reset()
        else:
            env.reset(change_sample=True)

        bw = env.link_available_bw
        bw = [1/(i+0.1) for i in bw]
        bw = 1+(10-1)*(bw-np.min(bw))/(np.max(bw)-np.min(bw))
        routing = dict()
        # 处理每一个流量需求
        for i in env.G.nodes():
            routing_t = dict()
            for j in env.G.nodes():
                if i == j:
                    routing_t[j] = [j]
                    continue
                # 训练300次
                graph = nx.adjacency_matrix(env.G).todense()
                action_list = dict()
                for a in range(action_size):
                    action_list[a] = np.nonzero(graph[a])[1]
                RL = QLearningAgent(obs_n=action_size, act_n=action_size, action=action_list,
                                    learning_rate=0.7, gamma=0.9, e_greed=0.1)

                demand = env.traffic_demand[i][j]
                for epi in range(50):
                    state = i
                    while state != j:
                        # 动作为下一跳交换机的序号，状态为当前交换机的序号
                        # 如果state和action之间没有链路，则不能选择该action
                        action = RL.sample(state)
                        # RL.Q[state][action] += 1000
                        # idx = env.G[state][action]['id']
                        # r = bw[idx]
                        r = demand
                        RL.learn(state, action, r, action, False)
                        # 状态更新
                        state = action
                    RL.learn(state, action, r, action, True)

                states = []
                state = i
                episode_count = 0
                while state != j:
                    states.append(state)
                    # 无需随机探索
                    action = RL.predict(state)

                    episode_count = episode_count + 1
                    if episode_count > 30:
                        action = RL.sample(state)

                    # 去除环路
                    if action in states:
                        t = states.pop()
                        while t != action:
                            t = states.pop()
                    # 状态更新
                    state = action


                states.append(state)
                routing_t[j] = states
                # print(states)
            routing[i] = routing_t

        # 存储结果
        # print(routing)
        routings.append(routing)
        reward = env.step(routing)
        print('eps',episode,' reward', reward)

        env_info, traffic, utilization, num_sample = env.get_env_info()
        save_statistic('QL_', env_info, traffic, utilization, num_sample, reward)
    return routings



if __name__ == "__main__":
    # NSFNet GEANT2 GBN
    # gravity_1 gravity_2 gravity_few uniform
    env = Environment(env_type='GBN', traffic_profile='gravity_1', init_sample=-1,
                      action_type='rsir', routing='sp')
    ACTION_SIZE = len(env.G.nodes)
    routings = train(ACTION_SIZE)

    # DDPGcompare(routings[0], routings[1])
    # statistics(routings, ACTION_SIZE)
