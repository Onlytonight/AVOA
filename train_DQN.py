import copy
import numpy as np
import DQN.agent as agent
from config.config import *
from env.environment import Environment
from util.util import save_statistic


def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):
        if i % 1 == 0:
            print('change demand')
            env.reset(change_sample=True)
        else:
            env.reset()

        episodic_reward = 0
        state = env.get_state()
        t = 0

        while t < max_t:
            t += 1
            action = agent.act(state, eps)
            # print(state)
            # print(action)
            if env.action_type == 'host-host':
                action_t = [_ // 7 for _ in action]
                action_t = np.array(action_t).reshape(14, 14)
            else:
                action_t = [_ + 1 for _ in action]
            # print(action_t)
            next_state, reward = env.step(action_t)
            env_info, traffic, utilization, num_sample = env.get_env_info()
            save_statistic('dqn', env_info, traffic, utilization, num_sample, reward)

            # next_state = next_state
            reward_t = [reward for i in range(ACTION_LEN)]
            agent.memory.append((state, action, reward_t, next_state))

            # 到了应该学习的时间和足够案例
            if t % 2 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()
                agent.soft_update(agent.tau)

            state = copy.deepcopy(next_state)
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        # 记录reward平均值，更方便观察
        average_log.append(np.mean(rewards_log[-100:]))
        # 每五十次换一行
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 50 == 0:
            print(np.sum(rewards_log))

        eps = max(eps * eps_decay, eps_min)

    return rewards_log


if __name__ == '__main__':
    env = Environment(routing='ecmp', action_type='all_link_weight', K=6, weigths_to_states=True,
                      link_traffic_to_states=True,
                      link_utilization_to_states=True, probs_to_states=False)
    print(len(env.G.edges))
    print(env.num_features)
    STATE_SIZE = env.num_features * len(env.G.edges)
    if env.action_type == 'host-host':
        ACTION_LEN = 14 * 14
    else:
        ACTION_LEN = len(env.G.edges)
    agent = agent.Agent(STATE_SIZE, ACTION_SIZE, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, ACTION_LEN)
    rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    # np.save('{}_rewards.npy'.format(MY_ENV), rewards_log)
    agent.Q_local.to('cpu')
    # torch.save(agent.Q_local.state_dict(), '{}_weights.pth'.format(MY_ENV))
