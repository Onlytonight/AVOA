import numpy as np
import PPO.agent as agent
from config.config_PPO import *
from env.environment import Environment
from util.util import save_statistic


def train(agent, env,change_sample, n_episode, n_update=4, update_frequency=1, max_t=100, scale=1):
    rewards_log = []
    average_log = []
    state_history = []
    action_history = []
    reward_history = []

    for i in range(1, n_episode+1):
        if i%10==0 and change_sample==True:
            print('change demand')
            env.reset(change_sample=True)
        else:
            env.reset()
        state = env.get_state()+1

        t = 0
        if len(state_history) == 0:
            state_history.append(list(state))
        else:
            state_history[-1] = list(state)
        episodic_reward = 0

        while t < max_t:
            action = agent.act(state)
            action_history.append(action)
            action=(action+100)*10

            action[action<=0]=0.1
            next_state, reward= env.step(action)

            env_info,traffic,utilization,num_sample=env.get_env_info()
            agent_info=agent.get_agent_info()
            # save_originaldata(agent_info+'-change_sample'+str(change_sample)+'-',env_info,traffic,utilization)
            save_statistic(agent_info+'-change_sample'+str(change_sample)+'-',env_info,traffic,utilization,num_sample,reward)

            episodic_reward += reward

            reward_history.append(reward * scale)
            state = next_state
            state_history.append(list(state))
            t+=1

        if i % update_frequency == 0:
            states, actions, old_mu, old_logstd, log_probs, rewards= agent.process_data(state_history, action_history, reward_history, 64)
            for _ in range(n_update):
                agent.learn(states, actions, old_mu, old_logstd, log_probs, rewards)
            state_history = []
            action_history = []
            reward_history = []

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))

        # print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_log[-1]), end='')
        print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_log[-1]))

        if i % 200 == 0:
            print()
            print('\nEpisode {} did not end'.format(i))

    return rewards_log, average_log

if __name__ == '__main__':

    # env = gym.make(RAM_CONTINUOUS_ENV_NAME)
    env=Environment()
    if AGENT_TYPE=='PPO':
        agent = agent.PPO_Agent(state_size=42,
                                 action_size=42,
                                 lr=LEARNING_RATE,
                                 beta=BETA,
                                 eps=EPS,
                                 tau=TAU,
                                 gamma=GAMMA,
                                 device=DEVICE,
                                 hidden=HIDDEN_CONTINUOUS,
                                 share=SHARE,
                                 mode=MODE,
                                 use_critic=CRITIC,
                                 normalize=NORMALIZE)


    rewards_log, _ = train(agent=agent,
                           env=env,
                           change_sample=True,
                           n_episode=RAM_NUM_EPISODE,
                           n_update=N_UPDATE,
                           update_frequency=UPDATE_FREQUENCY,
                           max_t=MAX_T,
                           scale=SCALE)
    np.save('result/{}_rewards.npy'.format(RAM_CONTINUOUS_ENV_NAME), rewards_log)