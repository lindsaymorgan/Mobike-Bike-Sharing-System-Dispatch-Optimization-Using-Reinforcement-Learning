import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import random

# hyper parameters
EPSILON = 0.9
GAMMA = 0.9
LR = 0.01
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 1000
BATCH_SIZE = 32

EPISODES = 2000
need = pd.read_csv('fake_4region_trip_20170510.csv')


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
        self.obs_init[:self.region_num] -= self.out_nums[0, :self.region_num]
        self.obs_init[-self.region_num-2:-2] -= self.out_nums[0, -self.region_num-2:-2]


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
        self.obs[-self.region_num-2:] += self.in_nums[self.t - 1, :self.region_num]

        reward = 0

        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态 若不合理则不采取任何操作
        if move + self.obs[-self.region_num-2+region] >= 0 and move <= self.obs[-1]:
            self.obs[-self.region_num-2+region] += move
            # 更新货车状态
            self.obs[-1] -= move  # 更新货车上的单车数
            self.obs[-2] = region  # 更新货车位置
            # 更新之前的动作历史
            self.obs[-self.region_num-2-1] = move  # 搬动的单车数
            self.obs[-self.region_num-2-2] = region  # 货车位置

        self.obs[-self.region_num-2:] -= self.out_nums[self.t, :self.region_num]
        reward = np.sum(self.obs[-self.region_num-2:-2][self.obs [-self.region_num-2:-2]< 0])
        self.obs[-self.region_num-2:-2][self.obs [-self.region_num-2:-2]< 0] = 0

        return np.append(self.obs, self.t), reward, done


class Net(nn.Module):
    def __init__(self, NUM_STATES):
        super(Net, self).__init__()

        EMB_SIZE = 30
        OTHER_SIZE = NUM_STATES  # fixme: update this value based on the input

        self.fc1 = nn.Linear(OTHER_SIZE + EMB_SIZE * 2, 256).cuda()
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 1).cuda()
        self.fc2.weight.data.normal_(0, 0.1)

        self.emb = nn.Embedding(NUM_STATES, EMB_SIZE).cuda()

    def forward(self, x: torch.cuda.FloatTensor, stations: torch.cuda.LongTensor):
        emb = self.emb(stations).flatten(start_dim=1)
        x = torch.cat([x, emb], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class Dqn():
    def __init__(self, NUM_STATES, NUM_ACTIONS, region_num,move_amount_limit):
        self.eval_net, self.target_net = Net(NUM_STATES), Net(NUM_STATES)
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.loss = nn.MSELoss()
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_STATES = NUM_STATES
        self.move_amount_limit = move_amount_limit
        self.region_num=region_num
        self.fig, self.ax = plt.subplots()

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 10 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, [action], [reward], next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        # notation that the function return the action's index nor the real action
        # EPSILON
        # state = torch.unsqueeze(torch.FloatTensor(state) ,0)

        # feasible action
        feasible_action = list()
        for action in range(self.NUM_ACTIONS):
            move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
            region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
            if move + state[-self.region_num-2+region] >= 0 and move <= state[-1]:
                feasible_action.append(action)

        if np.random.randn() <= EPSILON:
            tmp_value = list()
            # 对每个feasible action算value
            state_1= [j for i, j in enumerate(state) if i not in [self.region_num, self.region_num+2, 2*self.region_num+4]]
            state_2=[j for i, j in enumerate(state) if i in [self.region_num, self.region_num+2, 2*self.region_num+4]]
            for action in feasible_action:

                move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))

                x = torch.FloatTensor([np.concatenate([state_1, np.array([move])])]).cuda()  # 区块单车量，时刻t，搬运量
                station = torch.LongTensor([np.concatenate([state_2, np.array([region])])]).cuda()  # 货车所在区域编号，目的区域编号

                action_value = self.eval_net.forward(x, station)
                tmp_value.append(action_value.cpu().detach().numpy()[0])
            max_indice = [i for i, j in enumerate(tmp_value) if j == max(tmp_value)]  # 找最大index
            action = feasible_action[random.choice(max_indice)]  # 如果有多个index随机选一个，获得对应action

        else:
            action = random.choice(feasible_action)
        return action

    def predict(self, state):
        # notation that the function return the action's index nor the real action
        # EPSILON
        # feasible action
        feasible_action = list()
        for action in range(self.NUM_ACTIONS):
            move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
            region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
            if move + state[region] >= 0 and move <= state[-1]:
                feasible_action.append(action)

        tmp_x = list()
        tmp_y = list()
        # 对每个feasible action算value
        state_1 = [j for i, j in enumerate(state) if
                   i not in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
        state_2 = [j for i, j in enumerate(state) if
                   i in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
        for action in feasible_action:
            move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
            region = int(np.floor(action / (2 * self.move_amount_limit + 1)))

            tmp_x.append(np.concatenate([state_1, np.array([move])]))
            tmp_y.append(np.concatenate([state_2, np.array([region])]))

        x = torch.FloatTensor(tmp_x).cuda()
        station = torch.LongTensor(tmp_y).cuda()

        action_val = self.target_net.forward(x, station)


        max_indice = [i for i, j in enumerate([i[0] for i in action_val]) if j == np.max([i[0] for i in action_val])] # 找最大index
        action = feasible_action[random.choice(max_indice)]  # 如果有多个index随机选一个，获得对应action



        return action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        # 切取sars切片
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.NUM_STATES]).cuda()
        # note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, self.NUM_STATES:self.NUM_STATES + 1].astype(np.long)).cuda()
        batch_reward = torch.FloatTensor(batch_memory[:, self.NUM_STATES + 1: self.NUM_STATES + 2]).cuda()
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.NUM_STATES:]).cuda()

        x = torch.cat((batch_state[:, :4], batch_state[:, -2:]), 1)
        move = torch.FloatTensor([[i[0] % (2 * self.move_amount_limit + 1) - self.move_amount_limit] for i in
                                  batch_action.cpu().detach().numpy()]).cuda()
        x = torch.cat((x, move), 1)

        y = torch.LongTensor([[i] for i in batch_state[:, 4].cpu().detach().numpy()]).cuda()
        region = torch.LongTensor([[int(np.floor(i[0] / (2 * self.move_amount_limit + 1)))] for i in
                                   batch_action.cpu().detach().numpy()]).cuda()
        y = torch.cat((y, region), 1)

        q_eval = self.eval_net(x, y)
        # q_next = self.target_net(batch_next_state).detach()

        tmp_q_next = list()
        for i in batch_next_state:
            state = i.cpu().detach().numpy()
            feasible_action = list()
            for action in range(self.NUM_ACTIONS):
                move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
                if move + state[region] >= 0 and move <= state[-1]:
                    feasible_action.append(action)

            tmp_x = list()
            tmp_y = list()
            # 对每个feasible action算value
            a_t = state[: - 3]
            v_pos = state[- 3]
            bike_num_t = state[-2:]
            for action in feasible_action:
                move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))

                tmp_x.append(np.concatenate([a_t, bike_num_t, np.array([move])]))
                tmp_y.append(np.array([v_pos, region]))

            x = torch.FloatTensor(tmp_x).cuda()
            station = torch.LongTensor(tmp_y).cuda()

            action_val = self.target_net.forward(x, station)
            tmp_q_next.append(float(action_val.max(1)[0].max().cpu().detach().numpy()))

        q_next = torch.FloatTensor(tmp_q_next).cuda()

        # q_target = batch_reward + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = batch_reward + GAMMA * q_next

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # 评估 agent, 跑 5 个episode，总reward求平均
    def evaluate(self, env, render=False):
        eval_reward = []
        for i in range(1):
            obs = env.init()
            episode_reward = 0
            while True:
                action = self.predict(obs)  # 预测动作，只选最优动作
                state, reward, done = env.step(action)
                episode_reward += reward
                print(f"obs:{state[:-1]} action:{action} reward:{reward} reward_sum:{episode_reward} t:{state[-1]}")
                print(
                    f"region:{int(np.floor(action / (2 * self.move_amount_limit + 1)))} move:{action % (2 * self.move_amount_limit + 1) - self.move_amount_limit}",
                    file=open("output_action.txt", "a"))
                # if render:
                #     env.render()
                if done:
                    break
            eval_reward.append(episode_reward)
        return np.mean(eval_reward)


def main():
    env = Env(region_num=4, move_amount_limit=15, eps_num=5)
    NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
    NUM_STATES = env.region_num + 3  # MountainCar-v0: (2,)
    net = Dqn(NUM_STATES, NUM_ACTIONS, env.move_amount_limit)
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in range(EPISODES):
        state = env.init()
        step_counter = 0
        reward_sum = 0
        while True:
            step_counter += 1
            # env.render()
            action = net.choose_action(state)
            print("the action is {}".format(action))
            next_state, reward, done = env.step(action)
            net.store_trans(state, action, reward, next_state)
            reward_sum += reward

            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn()
                if done:
                    print("episode {}, the reward is {}".format(episode, round(reward_sum, 3)))
                    print(f"{round(reward_sum, 3)}", file=open("old_files/output_result-ver1.txt", "a"))

            if done:
                step_counter_list.append(step_counter)
                net.plot(net.ax, step_counter_list)
                break

            state = next_state
    print(net.evaluate(env))


if __name__ == '__main__':
    main()
