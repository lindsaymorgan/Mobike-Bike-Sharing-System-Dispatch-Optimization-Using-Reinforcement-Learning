import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import copy

# hyper parameters
# EPSILON = 0.85
GAMMA = 0.99
LR = 0.001
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32

EPISODES = 20000
need = pd.read_csv('real_4region_trip_20170510_5stage.csv')
ts=int(time.time())


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
        self.determine_init_state()
        self.obs_init = np.array(copy.deepcopy(self.init_state)+ [0, 0,0,0]+copy.deepcopy(self.init_state)+[0, 0])  # 各方格单车量+货车位置+货车上的单车量
        self.obs_init[-self.region_num-2:-2] -= self.out_nums[0, ]


    def init(self):
        self.obs = self.obs_init.copy()
        self.t = 0
        return np.append(self.obs, self.t)

    def determine_init_state(self):
        region_outheat=[ sum(x) for x in zip(*self.out_nums) ]
        rate=8500/sum(region_outheat)
        self.init_state=[np.ceil(i*rate) for i in region_outheat]



    def check_limit(self,state,action,t):  #对一个state(不含t,14位),action,t(当前)是否合法

        tmp_obs = copy.deepcopy(state)
        tmp_obs[:self.region_num + 2] = tmp_obs[-self.region_num - 2:]  # 更新状态
        tmp_obs[-self.region_num - 2:-2] += self.in_nums[int(t),]

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10


        if move + tmp_obs[-self.region_num - 2 + region] >= 0 and move <= tmp_obs[-1] \
                and (tmp_obs[-self.region_num - 2 + region] - self.out_nums[int(t+1), region]) * move <= 0:
            return False #合法动作
        else:
            return True   #非法动作

    def get_feasible_action(self,state):  #对一个state，给出合法的action组

        feasible_action = list()
        feasible_move=list()
        feasible_region=list()
        for action in range(self.action_dim):
            if not self.check_limit(state[:-1], action, state[-1]):
                feasible_action.append(action)
                move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
                feasible_move.append(move)
                feasible_region.append(region)
        return feasible_action,feasible_move,feasible_region


    def step(self, action):

        # 更新时间状态
        self.t += 1
        if self.check_limit(self.obs,action,self.t-1):   #若不合理则不采取任何操作 结束周期 回报设为大负数
            done=True
            reward=-100000
            return np.append(self.obs, self.t), reward, done

        elif self.t == self.episode_num-1:
            done = True
        else:
            done = False

        self.obs[:self.region_num+2]=self.obs[-self.region_num-2:]  #更新状态

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10

        # 更新单车分布状态
        # 处理上时段骑入
        self.obs[-self.region_num-2:-2] += self.in_nums[self.t - 1, ]

        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态
        # if self.check_limit(action,self.t):
        self.obs[-self.region_num-2+region] += move
        # 更新货车状态
        self.obs[-1] -= move  # 更新货车上的单车数
        self.obs[-2] = region  # 更新货车位置
        # 更新之前的动作历史
        self.obs[-self.region_num-2-1] = move  # 搬动的单车数
        self.obs[-self.region_num-2-2] = region  # 货车位置


        self.obs[-self.region_num-2:-2] -= self.out_nums[self.t, ]
        reward = np.sum(self.obs[-self.region_num - 2:-2][self.obs[-self.region_num - 2:-2] < 0])
        self.obs[-self.region_num-2:-2][self.obs [-self.region_num-2:-2]< 0] = 0

        return np.append(self.obs, self.t), reward, done


class Net(nn.Module):
    def __init__(self, NUM_STATES):
        super(Net, self).__init__()

        EMB_SIZE = 100
        OTHER_SIZE = NUM_STATES-2  # fixme: update this value based on the input

        self.fc1 = nn.Linear(OTHER_SIZE + EMB_SIZE * 4, 256).cuda()
        # self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 64).cuda()
        # self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64, 1).cuda()
        # self.fc3.weight.data.normal_(0, 0.1)
        self.m = nn.Dropout(p=0.2).cuda()

        self.emb = nn.Embedding(NUM_STATES, EMB_SIZE).cuda()

    def forward(self, x: torch.cuda.FloatTensor, stations: torch.cuda.LongTensor):
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


class Dqn():
    def __init__(self, NUM_STATES, NUM_ACTIONS, region_num,move_amount_limit, eps_num):
        self.eval_net, self.target_net = Net(NUM_STATES), Net(NUM_STATES)
        self.target_net.load_state_dict(self.eval_net.state_dict())
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
        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.eps_num=eps_num
        self.t_index = {i: str(i) for i in range(eps_num)}
        self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(eps_num)])

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 10 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, [action], [reward], next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state,EPSILON,env):
        # print(EPSILON)
        if random.random() > EPSILON:
            action=self.predict(state,env)

        else:
            feasible_action,m,r=env.get_feasible_action(state)
            action = random.choice(feasible_action)
        return action

    def predict(self, state,env):
        # notation that the function return the action's index nor the real action
        # EPSILON
        # feasible action
        state_1 = [j for i, j in enumerate(state) if
                   i not in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
        state_2 = [j for i, j in enumerate(state) if
                   i in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]

        tmp_x=list()
        tmp_y=list()
        feasible_action,m,r=env.get_feasible_action(state)
        for move,region in zip(m,r):
            tmp_x.append(np.concatenate([state_1, np.array([move])]))
            tmp_y.append(np.concatenate([state_2, np.array([region])]))

        x = torch.FloatTensor(tmp_x).cuda()
        station = torch.LongTensor(tmp_y).cuda()

        action_val = self.target_net.forward(x, station)

        max_indice = [i for i, j in enumerate([i[0] for i in action_val]) if
                      j == np.max([i[0] for i in action_val])]  # 找最大index
        action = feasible_action[random.choice(max_indice)]  # 如果有多个index随机选一个，获得对应action

        return action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def learn(self,env):
        # learn 100 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        # 切取sars切片
        batch_memory = self.memory[sample_index, :]
        batch_reward = torch.FloatTensor(batch_memory[:, self.NUM_STATES + 1: self.NUM_STATES + 2]).cuda()

        x=torch.FloatTensor(np.delete(batch_memory[:, :self.NUM_STATES],
                  [self.region_num,self.region_num+2,self.region_num*2+4], 1)).cuda()
        move = torch.FloatTensor([[(i[0] % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10] for i in
                                  batch_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        x = torch.cat((x, move), 1)

        y=torch.LongTensor(batch_memory[:, [self.region_num,self.region_num+2,self.region_num*2+4]]).cuda()
        region = torch.LongTensor([[int(np.floor(i[0] / (2 * self.move_amount_limit + 1)))] for i in
                                   batch_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        y = torch.cat((y, region), 1)

        q_eval = self.eval_net(x, y)

        tmp_q_next = list()
        for state in batch_memory[:, -self.NUM_STATES:]:
            if state[-1]!=self.eps_num-1:
                feasible_action, m, r = env.get_feasible_action(state)

                tmp_x = list()
                tmp_y = list()
                # 对每个feasible action算value
                state_1 = [j for i, j in enumerate(state) if
                           i not in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
                state_2 = [j for i, j in enumerate(state) if
                           i in [self.region_num, self.region_num + 2, 2 * self.region_num + 4]]
                for move,region in zip(m,r):

                    tmp_x.append(np.concatenate([state_1, np.array([move])]))
                    tmp_y.append(np.concatenate([state_2, np.array([region])]))

                x = torch.FloatTensor(tmp_x).cuda()
                station = torch.LongTensor(tmp_y).cuda()

                action_val = self.target_net.forward(x, station)
                tmp_q_next.append([float(action_val.max(1)[0].max().cpu().detach().numpy())])
            else:
                tmp_q_next.append([0])

        q_next = torch.FloatTensor(tmp_q_next).cuda()

        # q_target = batch_reward + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = batch_reward + GAMMA * q_next


        loss = self.loss(q_eval, q_target)
        # print(loss.item(), file=open(f"result_history/actionless_output_loss_{ts}.txt", "a"))
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
                obs, reward, done = env.step(action)
                episode_reward += reward
                print(f"obs:{obs[:-1]} action:{action} reward:{reward} reward_sum:{episode_reward} t:{obs[-1]}")
                print(
                    f"obs:{obs[:-1]} t:{obs[-1]} region:{int(np.floor(action / (2 * self.move_amount_limit + 1)))} "
                    f"move:{10*(action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)} reward:{reward} "
                    f"reward_sum:{episode_reward}",
                    file=open(f"result_action/actionless_output_action_{ts}.txt", "a"))
                # if render:
                #     env.render()
                if done:
                    break
            eval_reward.append(episode_reward)
        return np.mean(eval_reward)


def main():
    eps_num = 5
    EPSILON = 0.9
    EPS_DECAY = 0.99
    env = Env(region_num=58, move_amount_limit=50, eps_num=eps_num)
    NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
    NUM_STATES = 2*env.region_num + 7 # MountainCar-v0: (2,)

    net = Dqn(NUM_STATES, NUM_ACTIONS, env.region_num, env.move_amount_limit, eps_num)
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in range(EPISODES):
        state = env.init()
        step_counter = 0
        reward_sum = 0
        history_action=[]
        EPSILON = max(EPSILON * EPS_DECAY, 0.05)
        while True:
            step_counter += 1
            # env.render()
            action = net.choose_action(state,EPSILON,env)

            move = (action % (2 * env.move_amount_limit + 1) - env.move_amount_limit)*10
            region=int(np.floor(action / (2 * env.move_amount_limit + 1)))
            history_action.append((region,move))

            # print("the action is {}".format(action))
            next_state, reward, done = env.step(action)
            # print(next_state,reward)
            net.store_trans(state, action, reward, next_state)
            reward_sum += reward

            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn(env)


            if done:
                print("episode {}, the reward is {}, history action {}".format(episode, round(reward_sum, 3),history_action))
                print(f"{round(reward_sum, 3)}", file=open(f"result_history/real5stage_output_result_{ts}.txt", "a"))
                break

            state = next_state
    print(net.evaluate(env))


if __name__ == '__main__':
    main()
