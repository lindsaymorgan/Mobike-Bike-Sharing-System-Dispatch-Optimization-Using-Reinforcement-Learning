import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import copy
import pandas as pd
import random
import matplotlib.pyplot as plt


# 超参数
BATCH_SIZE = 32
EPISODES = 1000
LR = 0.01  # learning rate
EPSILON_MAX = 1  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆库大小
NODE_NUM = 34
N_STATES = 2
N_ACTIONS = NODE_NUM # total number of nodes


def build_karate_club_graph():
    g = nx.Graph()
    node_list = [i for i in range(34)]
    g.add_nodes_from(node_list)
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                 (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                 (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                 (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                 (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)]
    g.add_edges_from(edge_list)
    return g


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 20)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(20, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):  # 建立target net和eval net还有memory
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 用于target更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库；其中(N_STATES * 2 + 2)是s+a+r+s_总共的数量
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x, epsilon, executable_actions):  # 根据环境观测值选择动作的机制
        x = torch.unsqueeze(input=torch.FloatTensor(x), dim=0)  # 扩展维度，变为1xn的2维tensor（原x是1维的，包含n个元素）
        if np.random.uniform() < epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            action_sorted = torch.sort(input=actions_value, dim=1, descending=True)[1].data.numpy()[0]
            for a in action_sorted:
                if a in executable_actions:
                    action = a
                    break
            """
            torch.max(input, dim) 
                输入：input是一个tensor；dim是max索引的维度0/1，0表示每列的最大值，1表示每行的最大值
                输出：函数会返回两个tensor，第一个是对应的最大值，第二个是对应的最大值所在的索引

            所以这里的torch.max(actions_value, 1)[1]表示取Q值最大动作的索引值，然后用data.numpy()转为数据的形式；
            torch.sort()函数同理
            """
        else:  # 选随机动作
            action = np.random.choice(executable_actions)
        return action

    def store_transition(self, s, a, r, s_):  # 存储记忆
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了，就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):  # target网络更新；学习记忆库中的记忆
        # target_net参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 针对做过的动作b_a，来选q_eval的值。q_eval原本有所有动作的的值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape(batch, 1)
        q_next = self.target_net(b_s_).max(dim=1)[0].detach()  # q_next不进行反向传递误差，所以detach()——针对loss.backward()函数
        q_target = b_r + GAMMA * q_next.unsqueeze(1)  # shape(batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算，更新eval_net
        self.optimizer.zero_grad()  # 初始化梯度，重要
        loss.backward()  # 误差反向传递
        self.optimizer.step()  # 只有用了optimizer.step()，模型才会更新


def DQN_learning_percolation():
    dqn = DQN()  # 定义DQN系统

    for episode in range(1, EPISODES + 1):
        if episode % 100 == 0:
            print("learning episode = " + str(episode))

        EPSILON = episode / EPISODES * EPSILON_MAX
        percolation_list = []
        G = build_karate_club_graph()
        giant0 = len(max(nx.connected_components(G), key=len))
        removal0 = 0
        s = [giant0, removal0]
        executable_actions = [i for i in range(NODE_NUM)]

        while True:
            a = dqn.choose_action(s, EPSILON, executable_actions)

            # 选动作，得到反馈
            G.remove_node(a)
            executable_actions.remove(a)
            giant_ = len(max(nx.connected_components(G), key=len))
            removal_ = NODE_NUM - G.number_of_nodes()
            s_ = [giant_, removal_]
            r = (s[0] - s_[0]) / giant0
            percolation_list.append((a, giant_))

            if G.number_of_nodes() == 1:
                done = True
            else:
                done = False

            # 存记忆
            dqn.store_transition(s, a, r, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:  # 如果记忆库满了就进行学习
                dqn.learn()

            if done:  # 如果回合结束，进入下回合
                break

            s = copy.copy(s_)

        if episode % EPISODES == 0:
            df_rl = pd.DataFrame(percolation_list, columns=["node_index", "G"])
            df_rl.to_csv("karate-DQN percolation.csv", index=False)


def random_percolation():
    print("karate-random percolation")

    # 读入边列表
    G = build_karate_club_graph()

    node_list = []
    for v in G.nodes():
        node_list.append(v)
    random.shuffle(node_list)

    percolation_list = []
    for i in range(0, len(node_list) - 1):
        node = node_list[i]
        G.remove_node(node)

        largest_size = len(max(nx.connected_components(G), key=len))

        percolation_list.append((int(node), largest_size))

    df_random = pd.DataFrame(percolation_list, columns=["node_id", "G"])
    df_random.to_csv("karate-random percolation.csv", index=False)


def betweenness_percolation():
    print("karate-betweenness percolation")

    G = build_karate_club_graph()

    percolation_list = []
    while len(G.nodes()) > 1:
        node_bc_dict = nx.betweenness_centrality(G, k=None)
        node_bc_max = max(node_bc_dict.items(), key=lambda x: x[1])[0]  # 计算得到介数列表，选取介数最大项的边的连接信息即（u,v）

        G.remove_node(node_bc_max)

        largest_size = len(max(nx.connected_components(G), key=len))

        percolation_list.append((int(node_bc_max), largest_size))

    df_bc = pd.DataFrame(percolation_list, columns=['link_index', 'G'])
    df_bc.to_csv("karate-betweenness percolation.csv", index=False)


def plot_comparison():
    df_bc = pd.read_csv("karate-betweenness percolation.csv")
    df_ran = pd.read_csv("karate-random percolation.csv")
    df_rl = pd.read_csv("karate-DQN percolation.csv")
    df_rlql = pd.read_csv("karate-Q learning percolation.csv")
    x = [(i + 1) / len(df_bc) for i in range(len(df_bc))]

    plt.plot(x, df_bc['G'], c='black', linewidth=2, label='betweenness')
    plt.plot(x, df_ran['G'], c='gold', linestyle='--', linewidth=2, label='random')
    plt.plot(x, df_rl['G'], c='dodgerblue', linestyle='--', linewidth=2, label="DQN")
    plt.plot(x, df_rlql['G'], c='lime', linestyle='--', linewidth=2, label="Q learning")
    plt.xlabel('f', weight='bold', family='Arial', size=20)
    plt.ylabel('G', weight='bold', family='Arial', size=20)
    plt.xticks(weight='bold', family='Arial', size=15)
    plt.yticks(weight='bold', family='Arial', size=15)
    plt.legend(loc='best', fontsize=12)
    plt.title('karate-club network', weight='bold', family='Arial', size=15)
    plt.savefig("plot compare.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    #random_percolation()
    #betweenness_percolation()
    #DQN_learning_percolation()
    plot_comparison()