import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import scipy.stats as stats

need = pd.read_csv('fake_4region_trip_20170510.csv')
dist=pd.read_csv('fake_4region_distance.csv')
dist=dist.values
eps_num=5
car_num=2

class Env(object):
    def __init__(self, region_num, move_amount_limit, eps_num):
        self.region_num = region_num
        self.move_amount_limit = move_amount_limit
        self.action_dim = region_num * (2 * move_amount_limit + 1)
        self.car_num = car_num
        self.obs_dim = self.region_num + 2 * self.car_num
        self.episode_num = (eps_num-1)*car_num
        self.large_eps_num = eps_num
        self.current_car = 0
        self.current_eps = 0

        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.t_index = {i: str(i) for i in range(eps_num)}
        self.out_nums = np.array([need.groupby('start_region')[str(i)].agg(np.sum) for i in range(eps_num)])
        self.in_nums = np.array([need.groupby('end_region')[str(i)].agg(np.sum) for i in range(eps_num)])

        self.t = 0
        self.obs_init = np.array([15, 15, 15, 15, 0, 0, 0 ,0 ])  # 各方格单车量+货车位置+货车上的单车量
        self.obs_init[-self.region_num-4:-4] -= self.out_nums[0, ]

    def init(self):
        self.obs = self.obs_init.copy()
        self.t = 0
        return self.obs

    def check_limit(self,state,action, current_car, current_eps):  #对一个state,action,t(当前)是否合法

        tmp_obs = copy.deepcopy(state)

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit

        if move + tmp_obs[-self.obs_dim + region] >= 0 and move <= tmp_obs[-self.car_num+int(current_car)] \
                and (tmp_obs[-self.obs_dim + region] - self.out_nums[int(current_eps+1), region]) * move <= 0:
            return False #合法动作
        else:
            return True   #非法动作

    def calc_tmp_R(self):

        tmp_obs=self.obs.copy()
        tmp_obs[-self.obs_dim:-2 * self.car_num] -= self.out_nums[self.current_eps + 1,]
        raw_R=np.sum(tmp_obs[-self.obs_dim:-2 * self.car_num][tmp_obs[-self.obs_dim:-2 * self.car_num] < 0])

        return raw_R

    def step(self, action, obs,t,fore_R):

        # 当前决策周期+决策车辆
        self.current_car = self.t % self.car_num
        self.current_eps = int(np.floor(self.t / self.car_num))

        # 更新时间状态
        self.obs=obs

        self.t =t

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit

        # 更新单车分布状态
        # 处理上时段骑入
        if self.current_car == 0:
            self.obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(self.current_eps),]
            fore_R = self.calc_tmp_R()


        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态 若不合理则不采取任何操作
        if move + self.obs[-self.obs_dim + region] >= 0 and move <= self.obs[-self.car_num+int(self.current_car)]:
            self.obs[-self.obs_dim + region] += move
            # 更新货车状态
            self.obs[-self.car_num + self.current_car] -= move  # 更新货车上的单车数
            self.obs[-2 * self.car_num + self.current_car] = region  # 更新货车位置

        recent_R = self.calc_tmp_R()
        if self.current_car == 0:
            reward = recent_R
        else:
            reward = recent_R - fore_R

        if self.current_car == self.car_num - 1:
            self.obs[-self.obs_dim:-2 * self.car_num] -= self.out_nums[self.current_eps + 1,]
            self.obs[-self.obs_dim:-2 * self.car_num][self.obs[-self.obs_dim:-2 * self.car_num] < 0] = 0

        return tuple(np.append(self.obs, self.t)), reward, recent_R

env = Env(region_num=4, move_amount_limit=10, eps_num=5)
NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
NUM_STATES = 2 * env.region_num + 7  # MountainCar-v0: (2,)

history_dict={0: dict(),1: dict(),2: dict(),3:dict(),4:dict(),5: dict(),6: dict(),7: dict()}
history_action={0: dict(),1: dict(),2: dict(),3:dict(),4:dict(),5: dict(),6: dict(),7: dict()}

state = env.init()
print(state)

for action in range(NUM_ACTIONS):
    state = env.init()
    if not env.check_limit(state,action,0,0):
        fore_R = 0
        move = action % (2 * env.move_amount_limit + 1) - env.move_amount_limit
        region = int(np.floor(action / (2 * env.move_amount_limit + 1)))
        state,reward, fore_R=env.step(action,state,1,fore_R)
        if (state in history_dict[0] and history_dict[0][state] < reward) \
                or state not in history_dict[0]:
            history_dict[0][state] = (reward,fore_R)
            history_action[0][state] =(move,region,reward)



#逐个阶段
for stage in range(1,(eps_num-1)*car_num):
    current_car = stage % car_num
    current_eps = int(np.floor(stage / car_num))
    for action in range(NUM_ACTIONS):
        for state_fore,(reward_fore,fore_R) in history_dict[stage-1].items():
            if not env.check_limit(np.asarray(state_fore), action, current_car, current_eps):
                move = action % (2 * env.move_amount_limit + 1) - env.move_amount_limit
                region = int(np.floor(action / (2 * env.move_amount_limit + 1)))

                state, reward, fore_R = env.step(action, np.asarray(state_fore),stage+1,fore_R)
                if (state in history_dict[stage] and history_dict[stage][state][0] <reward+reward_fore)\
                        or state not in history_dict[stage]:
                    history_dict[stage][state] = (reward+reward_fore,fore_R)
                    history_action[stage][state] = (move,region,state_fore,reward)

max_value=max([i[0] for i in history_dict[7].values()])
max_state=[i for i,v in history_dict[7].items() if v[0]==max_value][0]

ts=int(time.time())
outfile=f"result_action/fakesmall_2cars_output_action_{ts}.txt"
reward_sum=0
for i in reversed(range(8)):
    print(max_state,history_action[i][max_state],
                        file=open(outfile, "a"))
    reward_sum+=history_action[i][max_state][-1]
    if i!=0:
        max_state=history_action[i][max_state][i-1]


print(f"best reward : {round(reward_sum/(eps_num-1), 3)}",
                    file=open(outfile, "a"))