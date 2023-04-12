import random
import time
import numpy as np

rng = np.random.default_rng()
import os
import numpy as np
import math
from env.environment import Environment
import operator


class Map:
    def __init__(self):
        self.t = -1
        self.iters = 1000
        # self.x_list = self.Chebyshevmap1(x0=0.7, iters=self.iters)
        # self.x_list = self.Circlemap2(0.5, 0.2, 0.7, iters=self.iters)
        self.x_list = self.tentMap(0.45)

    def Chebyshevmap1(self,x0, iters):
        x = x0
        x_list = []
        max_g = np.linspace(1, iters, num=iters)
        for i in max_g:
            x = math.cos(i * math.cos(x) ** (-1))
            x_list.append(x)
        return x_list

    def Circlemap2(self,a, b, x0, iters):
        a = a
        b = b
        x = x0
        x_list = []
        max_g = np.linspace(1, iters, num=iters)
        for i in max_g:
            x = (x + b - ((a / (2 * math.pi)) * math.sin(2 * math.pi * x))) % 1
            x_list.append(x)
        return x_list

    def tentMap(self,temp):
        Position = np.zeros(self.iters)
        for i in range(self.iters):
            if i == 0:
                Position[i] = random.random()
                print(Position[i])
            else:
                if Position[i - 1] < temp:
                    Position[i] = Position[i - 1] / temp
                else:
                    Position[i] = (1 - Position[i - 1]) / (1 - temp)
        return Position

    def getMap(self):
        if self.t < self.iters-1:
            self.t += 1
        else:
            self.t = 0
        # print(self.x_list[self.t],'  ',self.t)
        return abs(self.x_list[self.t])


def fun(X):
    # output = sum(np.square(X)) + random.random()
    action = X
    # 计算奖励
    next_state, reward = env.step(action)
    # env_info, traffic, utilization, num_sample = env.get_env_info()
    # save_statistic('AVOA-all-', env_info, traffic, utilization, num_sample, reward)
    env.reset()

    if math.isnan(action[0]):  # 某些情况下action为nan
        print(action)
    output = reward
    # print(reward)
    return output


def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))
    return x


# This function is to initialize the Vulture population.
def initial(pop, dim, ub, lb):
    between = env.betweeness()
    bd = env.bandwidth()
    bd = bd/(max(bd))
    sB = softmax(1 - between)

    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j] # 原始
            # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j] + (1 - between[j])
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j] + between[j]*beta  # +bd[j]*0.1
            # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]+ sB[j]
            # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]+(1-between[j]) + sB[j]
    print(X)
    return X


# Calculate fitness values for each Vulture
def CaculateFitness1(X, fun):
    fitness = fun(X)
    return fitness


# Sort fitness.
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


# Sort the position of the Vulture according to fitness.
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


# Boundary detection function.
def BorderCheck1(X, lb, ub, dim):
    for j in range(dim):
        if X[j] < lb[j]:
            X[j] = ub[j]
        elif X[j] > ub[j]:
            X[j] = lb[j]
    return X


def rouletteWheelSelection(x):
    CS = np.cumsum(x)
    Random_value = random.random()
    index = np.where(Random_value <= CS)
    index = sum(index)
    return index


def random_select(Pbest_Vulture_1, Pbest_Vulture_2, alpha, betha):
    probabilities = [alpha, betha]
    index = rouletteWheelSelection(probabilities)
    if (index.all() > 0):
        random_vulture_X = Pbest_Vulture_1
    else:
        random_vulture_X = Pbest_Vulture_2
    return random_vulture_X


def exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound, X, alp):
    if random.random() < p1:
        # eq6 两组最佳小组之一的随机距离附近随机搜索食物
        current_vulture_X = random_vulture_X - (abs((2 * random.random()) * random_vulture_X - current_vulture_X)) * F
    else:
        # eq8 搜索环境尺度上创建一个高随机系数
        current_vulture_X = (random_vulture_X - (F) + random.random() * (
                (upper_bound - lower_bound) * random.random() + lower_bound))
    # r1 = random.choice(range(0, pop))
    # r2 = random.choice(range(0, pop))
    # r3 = random.choice(range(0, pop))
    # current_vulture_X = X[r3] + random.random() * (X[r1] - X[r2])
    # current_vulture_X = random_vulture_X * (1 - alp) + ((X[r1] - X[r2]) * F + random_vulture_X) * alp
    return current_vulture_X


def exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no,
                 upper_bound, lower_bound):
    if abs(F) < 0.5:

        if random.random() < p2:
            # eq15 几种秃鹫在食物源上的聚集
            for item in current_vulture_X:
                if item == 0:
                    print(current_vulture_X)
                    break
            A = Best_vulture1_X - ((np.multiply(Best_vulture1_X, current_vulture_X)) / (
                    Best_vulture1_X - current_vulture_X ** 2)) * F
            # divide by zero encountered in true_divide
            B = Best_vulture2_X - (
                    (Best_vulture2_X * current_vulture_X) / (Best_vulture2_X - current_vulture_X ** 2)) * F
            current_vulture_X = (A + B) / 2
        else:
            # eq17 对食物的激烈竞争 朝着头秃鹫的不同方向移动
            current_vulture_X = random_vulture_X - abs(random_vulture_X - current_vulture_X) * F * levyFlight(
                variables_no)

    if random.random() >= 0.5:
        if random.random() < p3:
            # eq10 食物竞争
            current_vulture_X = (abs((2 * random.random()) * random_vulture_X - current_vulture_X)) * (
                    F + random.random()) - (random_vulture_X - current_vulture_X)

        else:
            # eq 12 and 13 旋转飞行
            s1 = random_vulture_X * (random.random() * current_vulture_X / (2 * math.pi)) * np.cos(current_vulture_X)
            s2 = random_vulture_X * (random.random() * current_vulture_X / (2 * math.pi)) * np.sin(current_vulture_X)
            current_vulture_X = random_vulture_X - (s1 + s2)
    return current_vulture_X


# eq (18) 莱维飞行
def levyFlight(d):
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / abs(v) ** (1 / beta)
    o = step
    return o


def save_reward(agent_info, iteration, reward):
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


def AVA(pop, dim, lb, ub, Max_iter, fun):
    alpha = 0.8
    betha = 0.2
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    Gama = 2.5
    X = initial(pop, dim, lb, ub)  # Initialize the random population
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = CaculateFitness1(X[i, :], fun)
    fitness, sortIndex = SortFitness(fitness)  # Sort the fitness values of African Vultures
    X = SortPosition(X, sortIndex)  # Sort the African Vultures population based on fitness
    GbestScore = fitness[0]  # Stores the optimal value for the current iteration.
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0, :]
    Curve = np.zeros([Max_iter, 1])
    Xnew = np.zeros([pop, dim])
    # Main iteration starts here
    for t in range(Max_iter+more_iter):
        Pbest_Vulture_1 = X[0, :]  # 选择最佳解作为第一组的最佳秃鹫 (First best location Best Vulture Category 1)
        Pbest_Vulture_2 = X[1, :]  # 选择次优解作为第二组的最佳秃鹫(Second best location Best Vulture Category 1)
        #
        t3 = np.random.uniform(-2, 2, 1) * (
                (np.sin((math.pi / 2) * (t / Max_iter)) ** Gama) + np.cos((math.pi / 2) * (t / Max_iter)) - 1)
        z = random.randint(-1, 1)
        F = (2 * random.random() + 1) * z * (1 - (t / Max_iter)) + t3
        P1 = (2 * random.random() + 1) * (1 - (t / Max_iter)) + t3
        # F = P1 * (2 * random.random() - 1)

        if t > Max_iter:
            # F = random.choice([0.1, -0.1])
            # t3 = np.random.uniform(-2, 2, 1) * (
            #         (np.sin((math.pi / 2) * (t / (Max_iter+more_iter))) ** Gama) + np.cos((math.pi / 2) * (t / (Max_iter+more_iter))) - 1)
            # F = (2 * random.random() + 1) * z * (1 - (t / (Max_iter+more_iter))) + t3
            F = 0
            # F = np.clip(F, -0.01, 0.01)

        fitenessList = []
        # For each vulture Pi
        for i in range(pop):
            current_vulture_X = X[i, :]
            random_vulture_X = random_select(Pbest_Vulture_1, Pbest_Vulture_2, alpha,
                                             betha)  # select random vulture using eq(1)
            if abs(F) >= 1:  # 探索阶段:提高种群多样性，有利于算法搜索最优解，但会降低收敛速率
                alp = np.cos(math.pi / 2 * (t / Max_iter))
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, ub, lb, X,
                                                alp)  # eq (16) & (17)

            else:  # 开发阶段:在最佳解的附近寻找食物，提高算法的收敛速度，但同时增加陷入早熟收敛的风险 早点进入开发阶段
                current_vulture_X = exploitation(current_vulture_X, Pbest_Vulture_1, Pbest_Vulture_2, random_vulture_X,
                                                 F, p2, p3, dim, ub, lb)  # eq (10) & (13)
            # print('nan----',current_vulture_X[0])
            Xnew[i, :] = current_vulture_X[0]
            Xnew[i, :] = BorderCheck1(Xnew[i, :], lb, ub, dim)
            tempFitness = CaculateFitness1(Xnew[i, :], fun)
            fitenessList.append(tempFitness)

            # Update local best solution
            if (tempFitness >= fitness[i]):
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]

        max_index, max_number = max(enumerate(fitenessList), key=operator.itemgetter(1))
        save_reward(path, t, max_number)

        Ybest, index = SortFitness(fitness)
        X = SortPosition(X, index)
        # Update global best solution
        if (Ybest[0] >= GbestScore):
            GbestScore = Ybest[0]
            GbestPositon[0, :] = X[index[0], :]
        # print(GbestPositon)

        # reward = GbestScore
        # env_info, traffic, utilization, num_sample = env.get_env_info()
        # save_statistic('AVOA-', env_info, traffic, utilization, num_sample, reward)
        env.reset(change_sample=True)
        print('reward', GbestScore, '  eps', t, "  F=", F)

        # Curve[t] = GbestScore
    return Curve, GbestPositon, GbestScore


myNum = Map()

ENV_TYPE = 'GBN'  # NSFNet GEANT2 GBN
TRAFFIC_PROFILE = 'gravity_1'  # gravity_1 gravity_2 gravity_few uniform
INIT_SAMPLE = 0
ROUTING = 'sp'
env = Environment(env_type=ENV_TYPE, traffic_profile=TRAFFIC_PROFILE, init_sample=INIT_SAMPLE, routing=ROUTING)

rng = np.random.default_rng()
time_start = time.time()
pop = 10  # Population size n
MaxIter = 5000  # Maximum number of iterations.
more_iter = 0

beta = 2

dim = len(env.G.edges)  # The dimension.
fl = 1  # The lower bound of the search interval.
ul = 2  # The upper bound of the search interval.搜索区间1-2
result_dir = 'result/pop'
path = str(ENV_TYPE) + '-' + str(TRAFFIC_PROFILE)+'pop='+str(pop)
lb = fl * np.ones([dim, 1])
ub = ul * np.ones([dim, 1])

Curve, GbestPositon, GbestScore = AVA(pop, dim, lb, ub, MaxIter, fun)  # Afican Vulture Optimization Algorithm
time_end = time.time()
print(f"The running time is: {time_end - time_start} s")
print('The optimal value：', GbestScore)
print('The optimal solution：', GbestPositon)

# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots()
# ax.plot(Curve, color='dodgerblue', marker='o', markeredgecolor='dodgerblue', markerfacecolor='dodgerblue')
# ax.set_xlabel('Number of Iterations', fontsize=15)
# ax.set_ylabel('Fitness', fontsize=15)
# ax.set_title('African Vulture Optimization')
# plt.savefig('result/img/image_'+ENV_TYPE+'_'+TRAFFIC_PROFILE+'_'+'static'+'.jpg', format='jpg')
# plt.show()
