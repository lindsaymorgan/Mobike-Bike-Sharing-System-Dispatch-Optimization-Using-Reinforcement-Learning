import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym
import pandas as pd

# Hyper Parameters
# STATE_DIM = 4
# ACTION_DIM = 2
STEP = 20000
SAMPLE_NUMS = 30
need = pd.read_csv('../fake_4region_trip_20170510.csv')

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


    def reset(self):
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
        # else:
        #     return False

        self.obs[-self.region_num-2:-2] -= self.out_nums[self.t, ]
        reward = np.sum(self.obs[-self.region_num-2:-2][self.obs [-self.region_num-2:-2]< 0])
        self.obs[-self.region_num-2:-2][self.obs [-self.region_num-2:-2]< 0] = 0

        return np.append(self.obs, self.t), reward, done

    def check(self,action):

        obs_tmp=self.obs
        obs_tmp[:self.region_num + 2] = obs_tmp[-self.region_num - 2:]  # 更新状态

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit

        # 更新单车分布状态
        # 处理上时段骑入
        obs_tmp[-self.region_num - 2:-2] += self.in_nums[self.t - 1,]

        reward = 0

        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态 若不合理则不采取任何操作
        if move + obs_tmp[-self.region_num - 2 + region] >= 0 and move <= self.obs[-1] \
                and (obs_tmp[-self.region_num - 2 + region] - self.out_nums[self.t, region]) * move <= 0:
            return True
        else:
            return False

class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class ValueNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def roll_out(actor_network,task,sample_nums,value_network,init_state):
    #task.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)
        log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)

        action = np.random.choice(task.action_dim,p=softmax_action.cpu().data.numpy()[0])
        while ~task.check(action):
            action = np.random.choice(task.action_dim, p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(task.action_dim)]
        next_state,reward,done = task.step(action)

        #fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = task.reset()
            break
    if not is_done:
        final_r = value_network(Variable(torch.Tensor([final_state]))).cpu().data.numpy()

    return states,actions,rewards,final_r,state

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    # init a task generator for data fetching
    task = Env(region_num=4, move_amount_limit=50, eps_num=5)
    ACTION_DIM = (2 * task.move_amount_limit + 1) * task.region_num  # [-500,500]*4个方块
    STATE_DIM = 2 * task.region_num + 7  # MountainCar-v0: (2,)
    init_state = task.reset()

    # init value network
    value_network = ValueNetwork(input_size = STATE_DIM,hidden_size = 256,output_size = 1)
    value_network_optim = torch.optim.Adam(value_network.parameters(),lr=0.01)

    # init actor network
    actor_network = ActorNetwork(STATE_DIM,256,ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.01)

    steps =[]
    task_episodes =[]
    test_results =[]

    for step in range(STEP):
        states,actions,rewards,final_r,current_state = roll_out(actor_network,task,SAMPLE_NUMS,value_network,init_state)
        init_state = current_state
        actions_var = Variable(torch.Tensor(actions).view(-1,ACTION_DIM))
        states_var = Variable(torch.Tensor(states).view(-1,STATE_DIM))

        # train actor network
        actor_network_optim.zero_grad()
        log_softmax_actions = actor_network(states_var)
        vs = value_network(states_var).detach()
        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))

        # advantages = qs - vs
        advantages = qs.view(-1,1) - vs
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        actor_network_optim.step()

        # train value network
        value_network_optim.zero_grad()
        target_values = qs
        values = value_network(states_var)
        criterion = nn.MSELoss()
        value_network_loss = criterion(values,target_values.view(-1, 1))
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(),0.5)
        value_network_optim.step()

        # Testing
        if (step + 1) % 50== 0:
                result = 0
                test_task = Env(region_num=4, move_amount_limit=50, eps_num=5)
                for test_epi in range(10):
                    state = test_task.reset()
                    # test_task.render()
                    for test_step in range(test_task.episode_num):
                        softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state]))))
                        #print(softmax_action.data)
                        # test_task.render()
                        action = np.argmax(softmax_action.data.numpy()[0])
                        next_state,reward,done = test_task.step(action)
                        result += reward
                        print(reward)
                        state = next_state
                        if done:
                            break
                print("step:",step+1,"test result:",result/10.0)
                steps.append(step+1)
                test_results.append(result/10)

if __name__ == '__main__':
    main()