import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import pandas as pd
import random
import time
import copy
import multiprocessing
from multiprocessing.pool import ThreadPool
import scipy.stats as stats
from tqdm import tqdm
from copy import deepcopy

# hyper parameters
# EPSILON = 0.85
GAMMA = 0.99
LR = 0.001
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32

EPISODES = 20000
need = pd.read_csv('fake_4region_trip_20170510.csv')
ts = int(time.time())

cores = multiprocessing.cpu_count() - 1
global pool
pool = ThreadPool(processes=cores)


class Env(object):
    def __init__(self, region_num, move_amount_limit, eps_num, car_num):
        self.region_num = region_num
        self.move_amount_limit = move_amount_limit
        self.action_dim = region_num * (2 * move_amount_limit + 1)
        self.car_num = car_num
        self.obs_dim = self.region_num + 2 * self.car_num
        self.episode_num = (eps_num - 1) * car_num
        self.large_eps_num = eps_num
        self.current_car = 0
        self.current_eps = 0

        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.t_index = {i: str(i) for i in range(eps_num)}
        self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(eps_num)])
        self.in_nums = np.array([self.end_region[str(i)].agg(np.sum) for i in range(eps_num)])

        self.t = 0

        self.obs_init = np.hstack(([15, 15, 15, 15], [0] * self.car_num, [0] * self.car_num, [0, 0], [15, 15, 15, 15],
                                   [0] * self.car_num, [0] * self.car_num))  # s:各方格单车量+货车位置+货车上的单车量 a:货车下一位置+货车搬运量
        self.obs_init[-self.obs_dim:-(2 * self.car_num)] -= self.out_nums[0,]

    def init(self):
        self.obs = self.obs_init.copy()
        self.t = 0
        return np.append(self.obs, self.t)

    def check_limit(self, arg):  # 对一个state(不含t,14位),action,t(当前)是否合法 Todo:check
        tmp_obs, action, current_car, current_eps = arg

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit

        # \ and (tmp_obs[-self.obs_dim + region] - self.out_nums[int(current_eps+1), region]) * move <= 0
        if move + tmp_obs[-self.obs_dim + region] >= 0 and move <= tmp_obs[-self.car_num + int(current_car)]:
            return False  # 合法动作
        else:
            return True  # 非法动作

    def get_feasible_action(self, state_with_t):  # 对一个state，给出合法的action组 Todo:check

        feasible_action = list()
        feasible_move = list()
        feasible_region = list()

        tmp_obs = copy.deepcopy(state_with_t[:-1])
        tmp_obs[:self.obs_dim] = tmp_obs[-self.obs_dim:]  # 更新状态

        current_car = (state_with_t[-1]) % self.car_num
        current_eps = np.floor((state_with_t[-1]) / self.car_num)
        # if current_car==0:
        #     tmp_obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(current_eps),]

        # result=pool.map(self.check_limit, [(tmp_obs, action, current_car, current_eps) for action in range(self.action_dim)])
        result = [self.check_limit((state_with_t[:-1], action, current_car, current_eps)) for action in
                  range(self.action_dim)]
        # result=[]
        # for action in range(self.action_dim):
        #     result.append(self.check_limit((tmp_obs, action, current_car, current_eps)))

        for action, r in enumerate(result):
            if not r:
                feasible_action.append(action)
                move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
                feasible_move.append(move)
                feasible_region.append(region)
        return feasible_action, feasible_move, feasible_region

    def calc_tmp_R(self):

        # tmp_obs=self.obs.copy()
        # tmp_obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(self.current_eps),]
        # # tmp_obs[-self.obs_dim:-2 * self.car_num] -= self.out_nums[self.current_eps + 1,]
        # raw_R = np.mean(
        #     [stats.poisson.cdf(i, j) for i, j in zip(tmp_obs[-self.obs_dim:-2 * self.car_num], self.out_nums[self.current_eps + 1,])])

        # tmp_obs = self.obs.copy()
        self.obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(self.current_eps),]
        # tmp_obs[-self.obs_dim:-2 * self.car_num] -= self.out_nums[self.current_eps + 1,]
        raw_R = np.mean(
            [stats.poisson.cdf(i, j) for i, j in
             zip(self.obs[-self.obs_dim:-2 * self.car_num], self.out_nums[self.current_eps + 1,])])
        self.obs[-self.obs_dim:-2 * self.car_num] -= self.in_nums[int(self.current_eps),]

        # raw_R=np.sum(tmp_obs[-self.obs_dim:-2 * self.car_num][tmp_obs[-self.obs_dim:-2 * self.car_num] < 0])

        return raw_R

    def step(self, action, fore_R):

        # 当前决策周期+决策车辆
        self.current_car = self.t % self.car_num
        self.current_eps = int(np.floor(self.t / self.car_num))

        # 更新时间状态
        self.t += 1

        # tmp_obs = copy.deepcopy(self.obs)
        # tmp_obs[:self.obs_dim] = tmp_obs[-self.obs_dim:]  # 更新状态
        # tmp_obs[-self.region_num - 2:-2] += self.in_nums[int(self.t-1),]
        # if self.check_limit((tmp_obs,action,self.t)):   #若不合理则不采取任何操作 结束周期 回报设为大负数
        #     done=True
        #     reward=-100000
        #     return np.append(self.obs, self.t), reward, done

        if self.t == self.episode_num:
            done = True
        else:
            done = False

        self.obs[:self.obs_dim] = self.obs[-self.obs_dim:]  # 更新状态

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = action % (2 * self.move_amount_limit + 1) - self.move_amount_limit

        # 更新单车分布状态
        # 处理上时段骑入

        # 计算初始R
        if self.current_car == 0:
            fore_R = self.calc_tmp_R()

        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态
        # if self.check_limit(action,self.t):
        self.obs[-self.obs_dim + region] += move
        # 更新货车状态
        self.obs[-self.car_num + self.current_car] -= move  # 更新货车上的单车数
        self.obs[-2 * self.car_num + self.current_car] = region  # 更新货车位置
        # 更新之前的动作历史
        self.obs[-self.obs_dim - 1] = move  # 搬动的单车数
        self.obs[-self.obs_dim - 2] = region  # 货车位置

        recent_R = self.calc_tmp_R()
        reward = recent_R - fore_R

        if self.current_car == self.car_num - 1:
            self.obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(self.current_eps),]
            self.obs[-self.obs_dim:-2 * self.car_num] -= self.out_nums[self.current_eps + 1,]
            self.obs[-self.obs_dim:-2 * self.car_num][self.obs[-self.obs_dim:-2 * self.car_num] < 0] = 0

        if self.current_car == 0:  # 如果是阶段起初则返回 原始R和当前R
            return np.append(self.obs, self.t), reward, fore_R, recent_R, done
        else:
            return np.append(self.obs, self.t), reward, recent_R, done


def weights_init_normal(layers, mean, std):
    for layer in layers:
        layer.weight.data.normal_(mean, std)


class Q_Net(nn.Module):
    def __init__(self, NUM_STATES, car_num, N_ACTIONS=1, H1Size=256, H2Size=64):
        super(Q_Net, self).__init__()

        EMB_SIZE = 10
        OTHER_SIZE = NUM_STATES + 2 - 2 * car_num - 2

        # build network layers
        self.fc1 = nn.Linear(OTHER_SIZE + EMB_SIZE * (2 * car_num + 2), H1Size).cuda()
        self.fc2 = nn.Linear(H1Size, H2Size).cuda()
        self.out = nn.Linear(H2Size, N_ACTIONS).cuda()
        self.emb = nn.Embedding(NUM_STATES, EMB_SIZE).cuda()

        # initialize layers
        weights_init_normal([self.fc1, self.fc2, self.out, self.emb], 0.0, 0.1)

    def forward(self, x: torch.cuda.FloatTensor, stations: torch.cuda.LongTensor):
        emb = self.emb(stations).flatten(start_dim=1)
        x = torch.cat([x, emb], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)

        return actions_value


class EnvModel(nn.Module):
    def __init__(self, NUM_STATES, car_num, N_ACTIONS=1, H1Size=256, H2Size=64):
        super(EnvModel, self).__init__()

        EMB_SIZE = 10
        OTHER_SIZE = NUM_STATES + 2 - 2 * car_num - 2

        # build network layers
        self.fc1 = nn.Linear(OTHER_SIZE + EMB_SIZE * (2 * car_num + 2), H1Size).cuda()
        self.fc2 = nn.Linear(H1Size, H2Size).cuda()
        self.statePrime = nn.Linear(H2Size, NUM_STATES).cuda()
        self.reward = nn.Linear(H2Size, 1).cuda()
        self.emb = nn.Embedding(NUM_STATES, EMB_SIZE).cuda()

        # initialize layers
        weights_init_normal([self.fc1, self.fc2, self.statePrime, self.reward, self.emb], 0.0, 0.1)

    # utils.weights_init_xavier([self.fc1, self.fc2, self.statePrime, self.reward, self.done])

    def forward(self, x: torch.cuda.FloatTensor, stations: torch.cuda.LongTensor):
        emb = self.emb(stations).flatten(start_dim=1)
        x = torch.cat([x, emb], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        statePrime_value = self.statePrime(x)
        reward_value = self.reward(x)

        return statePrime_value, reward_value


class DynaQ(object):
    def __init__(self, NUM_STATES, NUM_ACTIONS, region_num,move_amount_limit, eps_num,car_num):
        self.NUM_STATES = NUM_STATES
        self.NUM_ACTIONS = NUM_ACTIONS
        self.env_a_shape = 0


        self.eval_net = Q_Net(self.NUM_STATES, car_num )
        self.target_net = deepcopy(self.eval_net)
        self.env_model = EnvModel(self.NUM_STATES, car_num)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, self.NUM_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.env_opt = torch.optim.Adam(self.env_model.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

        self.move_amount_limit = move_amount_limit
        self.region_num=region_num
        self.car_num=car_num
        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.all_eps_num = (eps_num-1) * car_num
        self.eps_num=eps_num
        self.t_index = {i: str(i) for i in range(eps_num)}
        self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(eps_num)])


    def predict(self, state,env):
        indices1 = [i for i in range(self.region_num)] + \
            [i for i in range(self.region_num + self.car_num, self.region_num + 2 * self.car_num)] + \
            [self.region_num * 2 + self.car_num * 2 + 1] + \
            [i for i in range(self.region_num + 3 * self.car_num + 2, len(state))]# todo check ending
        state_1 = state[indices1]

        indices2 = [i for i in range(self.region_num, self.region_num + self.car_num)] + \
        [self.region_num + 2 * self.car_num] + \
        [i for i in range(2 * self.region_num + 2 * self.car_num + 2, 2 * self.region_num + 3 * self.car_num + 2)]
        state_2 = state[indices2]

        feasible_action,m,r=env.get_feasible_action(state)

        move = np.asarray(m)
        region = np.asarray(r)
        tmp = np.tile(state_1, (move.shape[0], 1))
        tmp_x = np.concatenate([tmp, move[:, np.newaxis]],axis=1)
        tmp = np.tile(state_2, (region.shape[0], 1))
        tmp_y = np.concatenate([tmp, region[:, np.newaxis]],axis=1)

        x = torch.FloatTensor(tmp_x).cuda()
        station = torch.LongTensor(tmp_y).cuda()

        action_val = self.target_net.forward(x, station)

        arr_action_val = action_val.detach().cpu().numpy().reshape(len(action_val))

        max_indice = np.argwhere(arr_action_val == np.amax(arr_action_val))[:, 0]

        action = feasible_action[random.choice(max_indice)]  # 如果有多个index随机选一个，获得对应action

        return action

    def choose_action(self, state, EPSILON, env):
        # input only one sample
        if np.random.uniform() > EPSILON:  # greedy
            action=self.predict(state, env)
        else:  # random
            feasible_action, m, r = env.get_feasible_action(state)
            action = random.choice(feasible_action)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def update_env_model(self):
        sample_index = np.random.choice(min(MEMORY_CAPACITY, self.memory_counter),BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # b_in = Variable(
        #     torch.FloatTensor(np.hstack((b_memory[:, :self.n_states], b_memory[:, self.n_states:self.n_states + 1]))))
        # b_y = Variable(torch.FloatTensor(np.hstack((b_memory[:, -self.n_states:], b_memory[:, self.n_states+1:self.n_states+2], b_memory[:, self.n_states+2:self.n_states+3]))))
        b_y_s = torch.FloatTensor(b_memory[:, -self.NUM_STATES:]).cuda()
        b_y_r = torch.FloatTensor(b_memory[:, self.NUM_STATES + 1:self.NUM_STATES + 2]).cuda()

        indices1 = [i for i in range(self.region_num)] + \
                   [i for i in range(self.region_num + self.car_num, self.region_num + 2 * self.car_num)] + \
                   [self.region_num * 2 + self.car_num * 2 + 1] + \
                   [i for i in range(self.region_num + 3 * self.car_num + 2, self.NUM_STATES)]
        x = torch.FloatTensor(b_memory[:,indices1]).cuda()

        indices2 = [i for i in range(self.region_num, self.region_num + self.car_num)] + \
                   [self.region_num + 2 * self.car_num] + \
                   [i for i in
                    range(2 * self.region_num + 2 * self.car_num + 2, 2 * self.region_num + 3 * self.car_num + 2)]
        y = torch.LongTensor(b_memory[:,indices2]).cuda()
        move = torch.FloatTensor([[i[0] % (2 * self.move_amount_limit + 1) - self.move_amount_limit] for i in
                                  b_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        x = torch.cat((x, move), 1)

        region = torch.LongTensor([[int(np.floor(i[0] / (2 * self.move_amount_limit + 1)))] for i in
                                   b_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        y = torch.cat((y, region), 1)


        b_s_, b_r = self.env_model(x, y)

        # loss = self.loss_func(torch.cat(b_out, 1), b_y)
        loss_s = self.loss_func(b_s_, b_y_s)
        loss_r = self.loss_func(b_r, b_y_r)

        self.env_opt.zero_grad()
        loss_s.backward(retain_graph=True)
        self.env_opt.step()

        self.env_opt.zero_grad()
        loss_r.backward()
        self.env_opt.step()

    def learn(self, env):
        # target parameter update
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(min(MEMORY_CAPACITY, self.memory_counter), BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        indices1 = [i for i in range(self.region_num)] + \
                   [i for i in range(self.region_num + self.car_num, self.region_num + 2 * self.car_num)] + \
                   [self.region_num * 2 + self.car_num * 2 + 1] + \
                   [i for i in range(self.region_num + 3 * self.car_num + 2, self.NUM_STATES)]
        x = torch.FloatTensor(b_memory[:,indices1]).cuda()

        indices2 = [i for i in range(self.region_num, self.region_num + self.car_num)] + \
                   [self.region_num + 2 * self.car_num] + \
                   [i for i in
                    range(2 * self.region_num + 2 * self.car_num + 2, 2 * self.region_num + 3 * self.car_num + 2)]
        y = torch.LongTensor(b_memory[:,indices2]).cuda()

        # b_s = Variable(torch.FloatTensor(b_memory[:, :self.NUM_STATES]))
        # b_a = Variable(torch.LongTensor(b_memory[:, self.NUM_STATES:self.NUM_STATES + 1].astype(int)))
        b_r = torch.FloatTensor(b_memory[:, self.NUM_STATES + 1:self.NUM_STATES + 2]).cuda()

        # q_eval w.r.t the action in experience
        # q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        move = torch.FloatTensor([[i[0] % (2 * self.move_amount_limit + 1) - self.move_amount_limit] for i in
                                  b_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        x = torch.cat((x, move), 1)

        # y=torch.LongTensor(state_2).cuda()
        region = torch.LongTensor([[int(np.floor(i[0] / (2 * self.move_amount_limit + 1)))] for i in
                                   b_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        y = torch.cat((y, region), 1)
        q_eval = self.eval_net(x, y)

        tmp_q_next = list()
        for state in b_memory[:, -self.NUM_STATES:]:
            if state[-1]!=self.all_eps_num:
                feasible_action, m, r = env.get_feasible_action(state)

                tmp_x = list()
                tmp_y = list()
                # 对每个feasible action算value

                indices1 = [i for i in range(self.region_num)] + \
                           [i for i in range(self.region_num + self.car_num, self.region_num + 2 * self.car_num)] + \
                           [self.region_num * 2 + self.car_num * 2 + 1] + \
                           [i for i in range(self.region_num + 3 * self.car_num + 2, len(state))]  # todo check ending
                state_1 = state[indices1]

                indices2 = [i for i in range(self.region_num, self.region_num + self.car_num)] + \
                           [self.region_num + 2 * self.car_num] + \
                           [i for i in range(2 * self.region_num + 2 * self.car_num + 2,
                                             2 * self.region_num + 3 * self.car_num + 2)]
                state_2 = state[indices2]
                # state_1 = np.delete(state,[i for i in range(self.region_num, self.region_num + self.car_num)] +
                #           [self.region_num + self.car_num * 2] +
                #           [i for i in range(self.region_num * 2 + self.car_num * 2 + 2,
                #                             self.region_num * 2 + self.car_num * 3 + 2)])
                #
                # state_2 = np.hstack((state[self.region_num:self.region_num + self.car_num],
                #                      [state[self.region_num + 2 * self.car_num]],
                #                      state[2 * self.region_num + 2 * self.car_num + 2:2 * self.region_num + 3 * self.car_num + 2]))

                for move,region in zip(m,r):

                    tmp_x.append(np.concatenate([state_1, np.array([move])]))
                    tmp_y.append(np.concatenate([state_2, np.array([region])]))

                x = torch.FloatTensor(tmp_x).cuda()
                station = torch.LongTensor(tmp_y).cuda()

                action_val = self.target_net.forward(x, station)
                tmp_q_next.append([float(action_val.max(1)[0].max().cpu().detach().numpy())])
            else:
                tmp_q_next.append([0])

        q_next = torch.FloatTensor(tmp_q_next).cuda().detach()

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
        # for param in self.eval_net.parameters():
        # 	param.grad.data.clamp_(-0.5, 0.5)
        self.optimizer.step()

    def simulate_learn(self, env):
        # target parameter update
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(min(MEMORY_CAPACITY, self.memory_counter), BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # b_s = b_memory[:, :self.n_states]
        indices1 = [i for i in range(self.region_num)] + \
                   [i for i in range(self.region_num + self.car_num, self.region_num + 2 * self.car_num)] + \
                   [self.region_num * 2 + self.car_num * 2 + 1] + \
                   [i for i in range(self.region_num + 3 * self.car_num + 2, self.NUM_STATES)]
        x = torch.FloatTensor(b_memory[:,indices1]).cuda()

        indices2 = [i for i in range(self.region_num, self.region_num + self.car_num)] + \
                   [self.region_num + 2 * self.car_num] + \
                   [i for i in
                    range(2 * self.region_num + 2 * self.car_num + 2, 2 * self.region_num + 3 * self.car_num + 2)]
        y = torch.LongTensor(b_memory[:,indices2]).cuda()
        move = torch.FloatTensor([[i[0] % (2 * self.move_amount_limit + 1) - self.move_amount_limit] for i in
                                  b_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        x = torch.cat((x, move), 1)

        region = torch.LongTensor([[int(np.floor(i[0] / (2 * self.move_amount_limit + 1)))] for i in
                                   b_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        y = torch.cat((y, region), 1)

        # # cartpole random generated data
        # b_s_s = np.random.uniform(low=-2.4, high=2.4, size=(self.config['batch_size'], 1))
        # b_s_theta = np.random.uniform(low=-0.2094, high=0.2094, size=(self.config['batch_size'], 1))
        # b_s_v = np.random.normal(scale=10, size=(self.config['batch_size'], 1))
        # b_s_w = np.random.normal(scale=10, size=(self.config['batch_size'], 1))
        # b_s = np.hstack((b_s_s, b_s_v, b_s_theta, b_s_w))

        # mountaincar random generated data
        # b_s_s = np.random.uniform(low=-1.2, high=0.6, size=(self.config['batch_size'], 1))
        # b_s_v = np.random.uniform(low=-0.07, high=0.07, size=(self.config['batch_size'], 1))
        # b_s = np.hstack((b_s_s, b_s_v))

        # b_a = np.random.randint(self.n_actions, size=b_s.shape[0])
        # b_a = np.reshape(b_a, (b_a.shape[0], 1))
        # b_in = Variable(torch.FloatTensor(np.hstack((b_s, np.array(b_a)))))

        statePrime_value, reward_value = self.env_model(x, y)
        # b_s = Variable(torch.FloatTensor(b_s))
        # b_a = Variable(torch.LongTensor(b_a))
        b_s_ = statePrime_value.detach()
        b_r = reward_value.detach()

        # q_eval w.r.t the action in experience
        # q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_eval = self.eval_net(x, y)

        # q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        # q_target = b_r + self.config['discount'] * q_next.max(1)[0].view(self.config['batch_size'],
        #                                                                  1) * b_d  # shape (batch, 1)
        tmp_q_next = list()
        for state in b_memory[:, -self.NUM_STATES:]:
            if state[-1]!=self.all_eps_num:
                feasible_action, m, r = env.get_feasible_action(state)

                tmp_x = list()
                tmp_y = list()
                # 对每个feasible action算value

                indices1 = [i for i in range(self.region_num)] + \
                           [i for i in range(self.region_num + self.car_num, self.region_num + 2 * self.car_num)] + \
                           [self.region_num * 2 + self.car_num * 2 + 1] + \
                           [i for i in range(self.region_num + 3 * self.car_num + 2, len(state))]  # todo check ending
                state_1 = state[indices1]

                indices2 = [i for i in range(self.region_num, self.region_num + self.car_num)] + \
                           [self.region_num + 2 * self.car_num] + \
                           [i for i in range(2 * self.region_num + 2 * self.car_num + 2,
                                             2 * self.region_num + 3 * self.car_num + 2)]
                state_2 = state[indices2]
                # state_1 = np.delete(state,[i for i in range(self.region_num, self.region_num + self.car_num)] +
                #           [self.region_num + self.car_num * 2] +
                #           [i for i in range(self.region_num * 2 + self.car_num * 2 + 2,
                #                             self.region_num * 2 + self.car_num * 3 + 2)])
                #
                # state_2 = np.hstack((state[self.region_num:self.region_num + self.car_num],
                #                      [state[self.region_num + 2 * self.car_num]],
                #                      state[2 * self.region_num + 2 * self.car_num + 2:2 * self.region_num + 3 * self.car_num + 2]))

                for move,region in zip(m,r):

                    tmp_x.append(np.concatenate([state_1, np.array([move])]))
                    tmp_y.append(np.concatenate([state_2, np.array([region])]))

                x = torch.FloatTensor(tmp_x).cuda()
                station = torch.LongTensor(tmp_y).cuda()

                action_val = self.target_net.forward(x, station)
                tmp_q_next.append([float(action_val.max(1)[0].max().cpu().detach().numpy())])
            else:
                tmp_q_next.append([0])

        q_next = torch.FloatTensor(tmp_q_next).cuda().detach()

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
        # for param in self.eval_net.parameters():
        # 	param.grad.data.clamp_(-0.5, 0.5)
        self.optimizer.step()

    def evaluate(self, env, render=False):
        eval_reward = []
        for i in range(1):
            obs = env.init()
            episode_reward = 0
            fore_R=0
            while True:
                action = self.predict(obs,env)  # 预测动作，只选最优动作
                if env.t%env.car_num!=0:
                    obs, reward, fore_R, done = env.step(action,fore_R)  #记录此阶段R 传入上一阶段R
                else:
                    next_state, reward, raw_R, fore_R, done = env.step(action, fore_R)
                    episode_reward += raw_R

                episode_reward += reward
                print(f"obs:{obs[:-1]} action:{action} reward:{reward} reward_sum:{episode_reward} t:{obs[-1]}")
                print(
                    f"obs:{obs[:-1]} t:{obs[-1]} region:{int(np.floor(action / (2 * self.move_amount_limit + 1)))} "
                    f"move:{action % (2 * self.move_amount_limit + 1) - self.move_amount_limit} reward:{reward} "
                    f"reward_sum:{episode_reward}",
                    file=open(f"result_action/pytorch_car2_output_action_{ts}.txt", "a"))
                # if render:
                #     env.render()
                if done:
                    break
            eval_reward.append(episode_reward)
        return np.mean(eval_reward)

def main():
    eps_num = 5
    car_num = 2
    EPSILON = 0.99
    EPS_DECAY = 0.999
    env = Env(region_num=4, move_amount_limit=3, eps_num=eps_num, car_num=car_num)
    NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
    NUM_STATES = 2 * env.region_num + 4 * car_num + 2 + 1  # 19

    net = DynaQ(NUM_STATES, NUM_ACTIONS, env.region_num, env.move_amount_limit, eps_num, car_num)
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in tqdm(range(EPISODES)):
        state = env.init()
        step_counter = 0
        reward_sum = 0
        history_action = []
        EPSILON = max(EPSILON * EPS_DECAY, 0.01)
        fore_R = 0
        while True:
            step_counter += 1
            # env.render()
            action = net.choose_action(state, EPSILON, env)

            move = action % (2 * env.move_amount_limit + 1) - env.move_amount_limit
            region = int(np.floor(action / (2 * env.move_amount_limit + 1)))
            history_action.append((region, move))

            if env.t % env.car_num == 0:
                next_state, reward, raw_R, fore_R, done = env.step(action, fore_R)
                reward_sum += raw_R
                net.store_transition(state, action, reward + raw_R, next_state)

            else:
                next_state, reward, fore_R, done = env.step(action, fore_R)
                net.store_transition(state, action, reward, next_state)

            # print(next_state,reward)
            # net.store_trans(state, action, reward, next_state)
            reward_sum += reward

            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn(env)
                # if done:
                #     print("episode {}, the reward is {}".format(episode, round(reward_sum, 3)))
                # print(f"{round(reward_sum, 3)}", file=open(f"result_history/actionless_output_result_{ts}.txt", "a"))

            if net.memory_counter > BATCH_SIZE:
                net.update_env_model()

            for _ in range(2):
                net.simulate_learn(env)

            if done:
                print("episode {}, the reward is {}, history action {}".format(episode,
                                                                               round(reward_sum / (eps_num - 1), 3),
                                                                               history_action))
                print(f"{round(reward_sum, 3) / (eps_num - 1)}",
                      file=open(f"result_history/smalltest_output_result_move_amount_limit{env.move_amount_limit}_{ts}",
                                "a"))
                break

            state = next_state

        if episode % 100 == 0:
            te = time.time()
            print(f'time consume: {te - ts}')
    print(net.evaluate(env))


if __name__ == '__main__':
    main()
