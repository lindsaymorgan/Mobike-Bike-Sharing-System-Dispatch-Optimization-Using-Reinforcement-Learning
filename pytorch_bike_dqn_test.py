import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import gym
import pandas as pd


#hyper parameters
EPSILON = 0.9
GAMMA = 0.9
LR = 0.01
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32

EPISODES = 400
need=pd.read_csv('4region_trip_20170510_eachhour.csv')
# env = gym.make('MountainCar-v0')
# env = env.unwrapped
# NUM_STATES = env.observation_space.shape[0] # 2
# NUM_ACTIONS = env.action_space.n

class Env(object):
    def __init__(self, region_num,move_amount_limit,eps_num):
        self.region_num=region_num
        self.move_amount_limit=move_amount_limit
        self.action_dim=region_num*(2*move_amount_limit+1)
        self.obs_dim=2*region_num+1
        self.t=0
        self.epsiode_num=eps_num
        self.obs=np.array([500,500,500,500,1,0,0,0,0]) #各方格单车量+货车位置+货车上的单车量
        out_num=np.array(need.groupby('start_region')[f'{self.t}'].agg(np.sum))
        for i in range(self.region_num):
            self.obs[i]-=out_num[i]


    def reset(self):
        self.obs = np.array([500, 500, 500, 500, 1, 0, 0, 0, 0])
        self.t=0
        out_num = np.array(need.groupby('start_region')[f'{self.t}'].agg(np.sum))
        for i in range(self.region_num):
            self.obs[i] -= out_num[i]
        return self.obs


    def step(self,action):

        # 更新时间状态
        self.t += 1
        if self.t == self.epsiode_num:
            done = True
        else:
            done = False
        _ = 0

        region=int(np.floor(action/(2*self.move_amount_limit+1)))
        move=action%(2*self.move_amount_limit+1)-self.move_amount_limit
        out_num = np.array(need.groupby('start_region')[f'{self.t}'].agg(np.sum))
        in_num = np.array(need.groupby('end_region')[f'{self.t - 1}'].agg(np.sum))
        if self.obs[-1]<0:
            print( self.obs[region], self.obs[-1], move)

        #更新单车分布状态
        for i in range(self.region_num):  #处理上时段骑入
            self.obs[i] += in_num[i]

        reward=0

        #筛选不合理情况 若合理 按照推算移动车辆 更新货车状态 若不合理则不采取任何操作
        if move + self.obs[region] >= 0 and move <= self.obs[-1]:
            self.obs[region] += move
            # 更新货车状态
            self.obs[-1] -= move  # 更新货车上的单车数
            for i in range(self.region_num, 2 * self.region_num):
                if self.obs[i] == 1:
                    self.obs[i] = 0
                    break
            self.obs[self.region_num + region] = 1  # 更新货车位置


        for i in range(self.region_num):

            if self.obs[i] >= out_num[i]:
                self.obs[i]-=out_num[i]

            #如果不能满足时间段内的所有需求
            else:
                reward+=(self.obs[i]-out_num[i]) #不能满足的部分设为损失
                self.obs[i]=0  #设余量为0

        return self.obs, reward, done, self.t

class Net(nn.Module):
    def __init__(self,NUM_STATES,NUM_ACTIONS):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(NUM_STATES, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(30, NUM_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class Dqn():
    def __init__(self,NUM_STATES,NUM_ACTIONS):
        self.eval_net, self.target_net = Net(NUM_STATES,NUM_ACTIONS), Net(NUM_STATES,NUM_ACTIONS)
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES *2 +2))
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.loss = nn.MSELoss()
        self.NUM_ACTIONS=NUM_ACTIONS
        self.NUM_STATES=NUM_STATES

        self.fig, self.ax = plt.subplots()

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 1000 ==0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, [action], [reward], next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        # notation that the function return the action's index nor the real action
        # EPSILON
        state = torch.unsqueeze(torch.FloatTensor(state) ,0)
        if np.random.randn() <= EPSILON:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy() # get action whose q is max
            action = action[0] #get the action index
        else:
            action = np.random.randint(0,self.NUM_ACTIONS)
        return action

    def predict(self, state):
        # notation that the function return the action's index nor the real action
        # EPSILON
        state = torch.unsqueeze(torch.FloatTensor(state) ,0)

        action_value = self.eval_net.forward(state)
        action = torch.max(action_value, 1)[1].data.numpy() # get action whose q is max
        action = action[0] #get the action index

        return action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter+=1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        #切取sars切片
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.NUM_STATES])
        #note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, self.NUM_STATES:self.NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.NUM_STATES+1: self.NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, net, render=False):
    eval_reward = []
    for i in range(1):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = net.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, t = env.step(action)
            episode_reward += reward
            print(f"obs:{obs} action:{action} reward:{reward} reward_sum:{episode_reward} t:{t}")
            # if render:
            #     env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():

    env = Env(region_num=4, move_amount_limit=500, eps_num=23)
    NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
    NUM_STATES = 2 * env.region_num + 1  # MountainCar-v0: (2,)
    net = Dqn(NUM_STATES, NUM_ACTIONS)
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in range(EPISODES):
        state = env.reset()
        step_counter = 0
        reward_sum=0
        while True:
            step_counter +=1
            # env.render()
            action = net.choose_action(state)
            next_state, reward, done, info = env.step(action)
            # reward = reward * 100 if reward >0 else reward * 5
            net.store_trans(state, action, reward, next_state)
            reward_sum+=reward

            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn()
                if done:
                    print("episode {}, the reward is {}".format(episode, round(reward_sum, 3)))
            if done:
                step_counter_list.append(step_counter)
                net.plot(net.ax, step_counter_list)
                break

            state = next_state
    print(evaluate(env,net))

if __name__ == '__main__':
    main()