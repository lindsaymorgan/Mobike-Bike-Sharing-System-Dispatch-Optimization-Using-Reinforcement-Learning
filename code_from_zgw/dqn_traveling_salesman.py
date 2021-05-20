import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 超参数
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆库大小
INF = 1000 # 无穷大用一个极大值表示
DIST_MAT = np.array([[INF, 3, INF, 8, 9], [3, INF, 3, 10, 5], [INF, 3, INF, 4, 3], [8, 10, 4, INF, 20], [9, 5, 3, 20, INF]])
N_ACTIONS = 5
N_STATES = 1


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_ACTIONS)
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

    def choose_action(self, x, executable_actions):  # 根据环境观测值选择动作的机制
        x = torch.unsqueeze(input=torch.FloatTensor(x), dim=0)  # 扩展维度，变为1xn的2维tensor（原x是1维的，包含n个元素）
        if np.random.uniform() < EPSILON:  # 选最优动作
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

            所以这里的torch.max(actions_value, 1)[1]表示取Q值最大动作的索引值，然后用data.numpy()转为数据的形式
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
        q_eval = self.eval_net(b_s).gather(dim=1, index=b_a)  # shape(batch, 1)
        """
        gather(dim, index), dim=0竖行查找，index的值表示行索引号；dim=1横向查找，index的值表示列索引号

        这里相当于把self.eval_net(b_s)返回的tensor（其shape为(batch, N_ACTIONS=2)），
        每一行（一共batch行）取某一个动作索引对应的值（这里是0或1），动作索引来源于index的tensor
        """
        q_next = self.target_net(b_s_).max(dim=1)[0].detach()  # q_next不进行反向传递误差，所以detach()——针对loss.backward()函数
        q_target = b_r + GAMMA * q_next.unsqueeze(1)  # shape(batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算，更新eval_net
        self.optimizer.zero_grad()  # 初始化梯度，重要
        loss.backward()  # 误差反向传递
        self.optimizer.step()  # 只有用了optimizer.step()，模型才会更新


if __name__ == "__main__":
    dqn = DQN()  # 定义DQN系统

    for episode in range(1, 1001):
        EPSILON = 1 / 1000 * episode # 重新设置EPSILON线性变化
        s = [0] # 从地点0出发
        executable_actions = [i for i in range(1, 5)]
        route = [0]
        for _ in range(200):
            a = dqn.choose_action(s, executable_actions)
            executable_actions.remove(a)
            route.append(a)

            # 选动作，得到反馈
            s_ = []
            s_.append(a)
            r = 1/DIST_MAT[s[0]][s_[0]]
            if len(executable_actions) == 0:
                r += 1/DIST_MAT[s_[0]][0]
                done = True
            else:
                done = False

            # 存记忆
            dqn.store_transition(s, a, r, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:  # 如果记忆库满了就进行学习
                dqn.learn()

            if done:  # 如果回合结束，进入下回合
                route.append(0)
                break

            s = s_

        cost = 0
        for i in range(len(route) - 1):
            cost += DIST_MAT[route[i]][route[i+1]]
        print('Episode:', episode, ', Route:', route, ', Cost:', cost)
