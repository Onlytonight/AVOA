import os
import numpy as np

def save_reward(result_dir,agent_info, iteration, reward):
    path = os.path.join(result_dir, agent_info)

    statistic_data = str(iteration) + ',' + str(reward)

    if not os.path.exists(path + '-statistic.csv'):
        with open(path + '-statistic.csv', 'w') as f:
            f.write(statistic_data + '\n')
            f.close()
    else:
        with open(path + '-statistic.csv', 'a') as f:
            f.write(statistic_data + '\n')
            f.close()


def save_statistic(agent_info,env_info,traffic,utilization,num_sample,reward):
    '''
    根据原始数据求出统计信息（t:traffic,u:utilization），包括tu上下四分位，tu上下界，tu中位数，tu均值，tu标准差，奖励，拥塞系数（链路利用率等于1的链路数在所有链路的占比），吞吐量，使用的主机流量
    t_lower_q,t_higher_q,t_median,t_max,t_min,t_mean,t_std(标准差),u_lower_q,u_higher_q,u_median,u_max,u_min,u_mean,u_s_deviation,congestion,throughput,reward
    :param agent_info: agent的信息  用于生成文件名
    :param env_info: 环境的信息  用于生成文件名
    :param traffic: 链路流量
    :param utilization:
    :param num_sample:
    :param reward:
    :return:
    '''
    headers='t_lower_q,t_higher_q,t_median,t_max,t_min,t_mean,t_std,u_lower_q,u_higher_q,u_median,u_max,u_min,u_mean,u_std,congestion,throughput,num_sample,reward'
    t_lower_q = np.quantile(traffic, 0.25, interpolation='lower')  # 下四分位数
    t_higher_q = np.quantile(traffic, 0.75, interpolation='higher')  # 上四分位数
    t_median=np.median(traffic)
    t_max=np.max(traffic)
    t_min=np.min(traffic)
    t_mean=np.mean(traffic)
    t_std=np.std(traffic)

    u_lower_q = np.quantile(utilization, 0.25, interpolation='lower')  # 下四分位数
    u_higher_q = np.quantile(utilization, 0.75, interpolation='higher')  # 上四分位数
    u_median=np.median(utilization)
    u_max=np.max(utilization)
    u_min=np.min(utilization)
    u_mean=np.mean(utilization)
    u_std=np.std(utilization)

    congestion = sum(i >= 1 for i in utilization) / len(utilization)

    throughput=np.sum(traffic)

    result_dir='result/compare'
    path = os.path.join(result_dir,agent_info+env_info)

    statistic_data = str(t_lower_q) + ',' + str(t_higher_q) + ',' + str(t_median) + ',' + str(t_max) + ',' + str(
        t_min) + ',' + str(t_mean) + ',' + str(t_std) + ',' + str(u_lower_q) + ',' + str(u_higher_q) + ',' + str(
        u_median) + ',' + str(u_max) + ',' + str(u_min) + ',' + str(u_mean) + ',' + str(u_std) + ',' + str(
        congestion) + ',' + str(throughput) + ',' + str(num_sample) + ',' + str(reward)


    if os.path.exists(path+'-statistic.csv') == False:
        with open(path+'-statistic.csv', 'w') as f:
            f.write(headers + '\n')
            f.write(statistic_data + '\n')
            f.close()
    else:
        with open(path+'-statistic.csv','a') as f:
            f.write(statistic_data + '\n')
            f.close()