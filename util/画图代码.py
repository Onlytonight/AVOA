import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import scipy.signal
from pylab import *
import csv
import os



def read_csv(filename,diskname,column):
    exampleFile = open(diskname + filename)  # 打开csv文件
    exampleReader = csv.reader(exampleFile)  # 读取csv文件
    exampleData = list(exampleReader)  # csv数据转换为列表
    length_zu = len(exampleData)  # 得到数据行数
    mpl.rcParams['axes.unicode_minus'] = False
    x = list()
    y = list()

    for i in range(header, length_zu):  # 从第二行开始读取
    # for i in range(0, length_zu):  # 从第二行开始读取
        x.append(float(exampleData[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
        y.append(float(exampleData[i][column]))  # 将第二列数据从第二行读取到最后一行赋给列表

    return x, y


def read_reward(diskname,column):
    filename_list = os.listdir(diskname)
    print(filename_list)
    a = []

    for i in filename_list:
        x, y = read_csv(i,diskname,column)
        reward = 0
        y1 = []
        for j in range(len(y)):
            reward += y[j]
            if j % r_eps == r_eps - 1:
                # reward /= 10
                y1.append(reward)
                reward = 0
        # y为reward列
        a.append(y1)

    return x, a, filename_list


def filter(y):
    a = np.empty(eps)
    for i in range(len(y)):
        a[i] = y[i]
    for i in range(len(y), eps):
        a[i] = 0
    return a


def smooth():
    # 平滑
    for i in range(len(y)):
        # y[i] = scipy.signal.savgol_filter(y[i], 100, 1)
        # y[i] = filter(y[i])
        # y[i][0]=-20
        for j in range(len(y[i])):
            y[i][j] = y[i][j]


def show(path, name):
    # plt.legend(fontsize=30)
    # plt.title('parameter-epsilon_decay', size=20)

    mpl.rcParams['font.sans-serif'] = ['SimHei']

    # 设置坐标轴范围
    plt.xlim((0, eps))
    plt.ylim((x_tick, y_tick))

    # 设置坐标轴名称，大小
    plt.xlabel('迭代次数', size=24)
    plt.ylabel('奖励', size=24)

    # 设置特定坐标轴刻度
    # 改变刻度
    my_x_ticks = np.arange(0, eps, scale)
    plt.xticks(my_x_ticks, size=30)
    my_y_ticks = np.arange(x_tick, y_tick, 1)
    plt.yticks(my_y_ticks, size=30)

    # 显示出所有设置
    plt.grid()
    plt.savefig(path+"/"+name)
    # plt.show()


def drawnormal():
    color = ['#FF9000', '#DE3325', '#00FF00', '#0085FF', '#9800FF', '#F19EBB', '#2B2B2B', '#2FC0DB', '#7591AE',
             '#2B2B2B']
    x = range(0, eps)
    for i in range(len(filename_list)):
        filename_list[i] = filename_list[i].split('-')
        # print(filename_list[i],np.std(y[i][1900:2000]))
        print(np.shape(y[i]))
        plt.plot(x, y[i], '-', color=color[i], label=filename_list[i][0], lw=1, zorder=20)


def drawsns(length):
    # 置信区间数据分组
    ys = []
    for i in range(eps):
        ys.append([i] * (length//eps))

    ys1 = []
    for i in range(10):
        ys1.append([(i + 1)] * 20)

    ys = np.asarray(ys).reshape(-1, )  # 100个数据为一组
    # print(y[0][-10000:])
    for i in range(len(filename_list)):
        filename_list[i] = filename_list[i].split('.')
        print(np.shape(y[i]))
        if len(y[i])<length:
            # print(np.mean(y[i][-10:]))
            # sns.lineplot(ys1, y[i][:], ci=95,label=filename_list[i][0])
            continue
        sns.lineplot(ys, y[i][-length:], ci=95,label=filename_list[i][0])
        # sns.lineplot(ys, y[i][0:5000], ci=95)


# diskname = r"D:\homework\网络\astppo\dqn_work\DQN+\result\AVOA_wrap\perfor/"
diskname = r"D:\homework\网络\astppo\dqn_work\DQN+\result\AVOA_wrap\5000/"
# diskname = r"D:\homework\网络\astppo\dqn_work\DQN+\result\AVOA_wrap\DDPGcompare/"
column = 1  # 绘制第几列数据
header = 0   # 文件首部
r_eps = 1 # 10个reward和
x, y, filename_list = read_reward(diskname,column)
# print(np.length(x))

# smooth()
plt.figure(num=10, figsize=(8, 8))
# eps = len(x)//r_eps
eps = 50  # 横坐标范围
x_tick =11# 纵坐标范围
y_tick = 20
scale = 10  # 刻度
length = 7000

path = "AVOA"
name = str(length)+'com'
drawnormal()
# drawsns(length)
show(path, name)