import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import random

# 超参数
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆库大小
need = pd.read_csv('../fake-very-small-test/fake_4region_trip_20170510.csv')

class Env(object):
    def __init__(self, region_num, move_amount_limit, eps_num):
        self.region_num = region_num
        self.move_amount_limit = move_amount_limit
        self.action_dim = region_num * (2 * move_amount_limit + 1)
        self.obs_dim = 2 * region_num + 1
        self.episode_num = eps_num

        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.t_index = {i: str(i) for i in range(eps_num)}
        self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(eps_num)])
        self.in_nums = np.array([self.end_region[str(i)].agg(np.sum) for i in range(eps_num)])

        self.t = 0
        self.obs_init = np.array([15, 15, 15, 15, 0, 0,0,0,15, 15, 15, 15, 0, 0])  # 各方格单车量+货车位置+货车上的单车量
        self.obs_init[-self.region_num-2:-2] -= self.out_nums[0, ]


    def init(self):
        self.obs = self.obs_init.copy()
        self.t = 0
        return np.append(self.obs, self.t)

    def step(self, action):

        # 更新时间状态
        self.t += 1
        if self.t == self.episode_num-1:
            done = True
        else:
            done = False

        self.obs[:self.region_num+2]=self.obs[-self.region_num-2:]  #更新状态

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit

        # 更新单车分布状态
        # 处理上时段骑入
        self.obs[-self.region_num-2:-2] += self.in_nums[self.t - 1, ]

        reward = 0

        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态 若不合理则不采取任何操作
        if move + self.obs[-self.region_num-2+region] >= 0 and move <= self.obs[-1] \
                and (self.obs[-self.region_num-2+region]- self.out_nums[self.t,region])*move<=0:
            self.obs[-self.region_num-2+region] += move
            # 更新货车状态
            self.obs[-1] -= move  # 更新货车上的单车数
            self.obs[-2] = region  # 更新货车位置
            # 更新之前的动作历史
            self.obs[-self.region_num-2-1] = move  # 搬动的单车数
            self.obs[-self.region_num-2-2] = region  # 货车位置

        self.obs[-self.region_num-2:-2] -= self.out_nums[self.t, ]
        reward = np.sum(self.obs[-self.region_num-2:-2][self.obs [-self.region_num-2:-2]< 0])
        self.obs[-self.region_num-2:-2][self.obs [-self.region_num-2:-2]< 0] = 0

        return np.append(self.obs, self.t), reward, done

class Net(nn.Module):
    def __init__(self, NUM_STATES):
        super(Net, self).__init__()

        EMB_SIZE = 10
        OTHER_SIZE = NUM_STATES - 2

        self.fc1 = nn.Linear(OTHER_SIZE + EMB_SIZE * 4, 256).cuda()
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(256, 64).cuda()
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64, 1).cuda()
        self.fc3.weight.data.normal_(0, 0.1)

        self.emb = nn.Embedding(NUM_STATES, EMB_SIZE).cuda()

    def forward(self, x,stations):
        emb = self.emb(stations).flatten(start_dim=1)
        x = torch.cat([x, emb], 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.m(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.m(x)
        x = self.fc3(x)
        return x


class DQN(object):
    def __init__(self, NUM_STATES, NUM_ACTIONS, region_num,move_amount_limit, eps_num):  # 建立target net和eval net还有memory
        self.eval_net, self.target_net = Net(NUM_STATES), Net(NUM_STATES)

        self.learn_step_counter = 0  # 用于target更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))  # 初始化记忆库；其中(N_STATES * 2 + 2)是s+a+r+s_总共的数量
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch的优化器
        self.loss= nn.MSELoss()  # 误差公式
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_STATES = NUM_STATES
        self.move_amount_limit = move_amount_limit
        self.region_num = region_num
        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.t_index = {i: str(i) for i in range(eps_num)}
        self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(eps_num)])

    def choose_action(self, state,EPSILON):  # 根据环境观测值选择动作的机制
        if random.random() > EPSILON:
            action = self.predict(state)

        else:
            feasible_action = list()
            for action in range(self.NUM_ACTIONS):
                move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
                if move + state[-self.region_num - 2 + region] >= 0 and move <= state[-2] and \
                        (state[-self.region_num - 2 + region] - self.out_nums[state[-1], region]) * move <= 0:
                    feasible_action.append(action)
            action = random.choice(feasible_action)
        return action

    def store_trans(self, s, a, r, s_):  # 存储记忆
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了，就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def predict(self, state):
        # notation that the function return the action's index nor the real action
        # EPSILON
        # feasible action
        feasible_action = list()
        state_1 = [j for i, j in enumerate(state) if
                   i not in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
        state_2 = [j for i, j in enumerate(state) if
                   i in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]

        tmp_x = list()
        tmp_y = list()
        for action in range(self.NUM_ACTIONS):
            move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
            region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
            if move + state[-self.region_num - 2 + region] >= 0 and move <= state[-2]\
                    and (state[-self.region_num-2+region]- self.out_nums[state[-1],region])*move<=0:
                feasible_action.append(action)
                tmp_x.append(np.concatenate([state_1, np.array([move])]))
                tmp_y.append(np.concatenate([state_2, np.array([region])]))

        x = torch.FloatTensor(tmp_x).cuda()
        station = torch.LongTensor(tmp_y).cuda()

        action_val = self.target_net.forward(x, station)

        max_indice = [i for i, j in enumerate([i[0] for i in action_val]) if
                      j == np.max([i[0] for i in action_val])]  # 找最大index
        action = feasible_action[random.choice(max_indice)]  # 如果有多个index随机选一个，获得对应action

        return action

    def learn(self):  # target网络更新；学习记忆库中的记忆
        # target_net参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # 切取sars切片
        batch_memory = self.memory[sample_index, :]
        batch_reward = torch.FloatTensor(batch_memory[:, self.NUM_STATES + 1: self.NUM_STATES + 2]).cuda()

        x = torch.FloatTensor(np.delete(batch_memory[:, :self.NUM_STATES],
                                        [self.region_num, self.region_num + 2, self.region_num * 2 + 4], 1)).cuda()
        move = torch.FloatTensor([[i[0] % (2 * self.move_amount_limit + 1) - self.move_amount_limit] for i in
                                  batch_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        x = torch.cat((x, move), 1)

        y = torch.LongTensor(batch_memory[:, [self.region_num, self.region_num + 2, self.region_num * 2 + 4]]).cuda()
        region = torch.LongTensor([[int(np.floor(i[0] / (2 * self.move_amount_limit + 1)))] for i in
                                   batch_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        y = torch.cat((y, region), 1)

        q_eval = self.eval_net(x, y)

        tmp_q_next = list()
        for state in batch_memory[:, -self.NUM_STATES:]:
            feasible_action = list()
            m_r_list = list()
            for action in range(self.NUM_ACTIONS):
                move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
                if move + state[-self.region_num - 2 + region] >= 0 and move <= state[-2] \
                        and (state[-self.region_num - 2 + region] - self.out_nums[int(state[-1]), region]) * move <= 0:
                    feasible_action.append(action)
                    m_r_list.append((move, region))

            tmp_x = list()
            tmp_y = list()
            # 对每个feasible action算value
            state_1 = [j for i, j in enumerate(state) if
                       i not in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
            state_2 = [j for i, j in enumerate(state) if
                       i in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
            for move, region in m_r_list:
                # move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
                # region = int(np.floor(action / (2 * self.move_amount_limit + 1)))

                tmp_x.append(np.concatenate([state_1, np.array([move])]))
                tmp_y.append(np.concatenate([state_2, np.array([region])]))

            x = torch.FloatTensor(tmp_x).cuda()
            station = torch.LongTensor(tmp_y).cuda()

            action_val = self.target_net.forward(x, station)
            tmp_q_next.append([float(action_val.max(1)[0].max().cpu().detach().numpy())])

        q_next = torch.FloatTensor(tmp_q_next).cuda()

        # q_target = batch_reward + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = batch_reward + GAMMA * q_next

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




if __name__ == "__main__":
    eps_num = 5
    EPSILON = 0.9
    EPS_DECAY = 0.99
    env = Env(region_num=4, move_amount_limit=10, eps_num=eps_num)
    NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
    NUM_STATES = 2 * env.region_num + 7  # MountainCar-v0: (2,)
    dqn = DQN(NUM_STATES, NUM_ACTIONS, env.region_num, env.move_amount_limit, eps_num)  # 定义DQN系统

    for episode in range(20000):
        EPSILON = min(1 / 1000 * episode,0.9) # 重新设置EPSILON线性变化
        state = env.init()
        reward_sum = 0
        while True:
            action = dqn.choose_action(state,EPSILON)

            # 选动作，得到反馈
            next_state, reward, done = env.step(action)
            dqn.store_trans(state, action, reward, next_state)
            reward_sum += reward

            if dqn.memory_counter > MEMORY_CAPACITY:  # 如果记忆库满了就进行学习
                dqn.learn()

            if done:  # 如果回合结束，进入下回合
                print("episode {}, the reward is {}".format(episode, round(reward_sum, 3)))
                break

            state = next_state
