import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
from tqdm import tqdm

need = pd.read_csv('../4region_trip_20170510.csv')
eps_num=4
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
        self.out_nums = np.array([need.groupby('start_region')[str(i)].agg(np.sum) for i in range(eps_num)])
        self.in_nums = np.array([need.groupby('end_region')[str(i)].agg(np.sum) for i in range(eps_num)])

        self.t = 0
        self.obs_init = np.array([500, 500, 500, 500, 0, 0])  # 各方格单车量+货车位置+货车上的单车量
        self.obs_init[-self.region_num-2:-2] -= self.out_nums[0, ]

    def init(self):
        self.obs = self.obs_init.copy()
        self.t = 0
        return self.obs

    def step(self, action, obs,t):
        # 更新时间状态
        self.obs=obs

        self.t =t

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10

        # 更新单车分布状态
        # 处理上时段骑入
        self.obs[-self.region_num - 2:-2] += self.in_nums[self.t - 1,]


        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态 若不合理则不采取任何操作
        if move + self.obs[-self.region_num - 2 + region] >= 0 and move <= self.obs[-1]:
            self.obs[-self.region_num - 2 + region] += move
            # 更新货车状态
            self.obs[-1] -= move  # 更新货车上的单车数
            self.obs[-2] = region  # 更新货车位置


        self.obs[-self.region_num - 2:-2] -= self.out_nums[self.t,]
        reward = np.sum(self.obs[-self.region_num - 2:-2][self.obs[-self.region_num - 2:-2] < 0])
        self.obs[-self.region_num - 2:-2][self.obs[-self.region_num - 2:-2] < 0] = 0

        return tuple(self.obs), reward

    def check_limit(self,state,action,t):  #对一个state(不含t,14位),action,t(当前)是否合法

        tmp_obs = copy.deepcopy(state)
        tmp_obs[:self.region_num + 2] = tmp_obs[-self.region_num - 2:]  # 更新状态

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10

        tmp_obs[-self.region_num - 2:-2] += self.in_nums[int(t),]

        if move + tmp_obs[-self.region_num - 2 + region] >= 0 and move <= tmp_obs[-1] \
                and (tmp_obs[-self.region_num - 2 + region] - self.out_nums[int(t+1), region]) * move <= 0:
            return False #合法动作
        else:
            return True   #非法动作

env = Env(region_num=4, move_amount_limit=50, eps_num=5)
NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
NUM_STATES = 2 * env.region_num + 7  # MountainCar-v0: (2,)

history_dict={0: dict(),1: dict(),2: dict(),3:dict(),4:dict()}
history_action={0: dict(),1: dict(),2: dict(),3:dict(),4:dict()}

state = env.init()
print(state)
for action in range(NUM_ACTIONS):
    state = env.init()

    if not env.check_limit(state,action,0):
        move = (action % (2 * env.move_amount_limit + 1) - env.move_amount_limit) * 10
        region = int(np.floor(action / (2 * env.move_amount_limit + 1)))
        state,reward=env.step(action,state,1)
        if (state in history_dict[0] and history_dict[0][state] < reward) \
                or state not in history_dict[0]:
            history_dict[0][state] =reward
            history_action[0][state] =(move,region,reward)



#逐个阶段
for stage in range(1,eps_num):
    for action in tqdm(range(NUM_ACTIONS)):
        for state_fore,reward_fore in history_dict[stage-1].items():

            if not env.check_limit(np.asarray(state_fore), action, stage):
                move = (action % (2 * env.move_amount_limit + 1) - env.move_amount_limit)*10
                region = int(np.floor(action / (2 * env.move_amount_limit + 1)))
                state, reward = env.step(action, np.asarray(state_fore),stage+1)

                if (state in history_dict[stage] and history_dict[stage][state] <reward+reward_fore)\
                        or state not in history_dict[stage]:
                    history_dict[stage][state] = reward+reward_fore
                    history_action[stage][state] = (move,region,state_fore,reward)

outfile=f"result_action/real500_output_action_{ts}.txt"
reward_sum=0
max_value=max(history_dict[3].values())
max_state_3=[i for i,v in history_dict[3].items() if v==max_value][0]
print(max_state_3,history_action[3][max_state_3],
                    file=open(outfile, "a"))
reward_sum+=history_action[3][max_state_3][-1]

max_state_2=history_action[3][max_state_3][2]
print(max_state_2,history_action[2][max_state_2],
                    file=open(outfile, "a"))
reward_sum+=history_action[2][max_state_2][-1]

max_state_1=history_action[2][max_state_2][2]
print(max_state_1,history_action[1][max_state_1],
                    file=open(outfile, "a"))
reward_sum+=history_action[1][max_state_1][-1]

max_state_0=history_action[1][max_state_1][2]
print(max_state_0,history_action[0][max_state_0],
                    file=open(outfile, "a"))
reward_sum+=history_action[0][max_state_0][-1]

print(f"best reward : {reward_sum}",
                    file=open(outfile, "a"))