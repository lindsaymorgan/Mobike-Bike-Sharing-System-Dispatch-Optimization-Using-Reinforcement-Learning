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
import multiprocessing
from multiprocessing.pool import ThreadPool
import scipy.stats as stats

# hyper parameters
# EPSILON = 0.85
GAMMA = 0.99
LR = 0.001
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32

EPISODES = 20000
need = pd.read_csv('../real_4region_trip_20170510_5stage.csv')
ts=int(time.time())

cores = multiprocessing.cpu_count()-1
global pool
pool = ThreadPool(processes=cores)


class Env(object):
    def __init__(self, region_num, move_amount_limit, eps_num,car_num):
        self.region_num = region_num
        self.move_amount_limit = move_amount_limit
        self.action_dim = region_num * (2 * move_amount_limit + 1)
        self.car_num = car_num
        self.obs_dim = self.region_num + 2 * self.car_num
        self.episode_num = (eps_num-1)*car_num
        self.large_eps_num = eps_num
        self.current_car = 0
        self.current_eps = 0

        # self.start_region = need.groupby('start_region')
        # self.end_region = need.groupby('end_region')
        # self.t_index = {i: str(i) for i in range(eps_num)}
        # self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(eps_num)])
        # self.in_nums = np.array([self.end_region[str(i)].agg(np.sum) for i in range(eps_num)])

        self.t = 0
        self.determine_init_state()
        self.obs_init = np.hstack((copy.deepcopy(self.init_state), [0]*self.car_num, [0]*self.car_num,[0,0],copy.deepcopy(self.init_state), [0]*self.car_num, [0]*self.car_num))# s:各方格单车量+货车位置+货车上的单车量 a:货车下一位置+货车搬运量
        self.obs_init[-self.obs_dim:-(2*self.car_num)] -= self.out_nums[0, ]

    def possion_flow_init(self):
        self.need = need.copy()
        for i in range(self.episode_num):  # 生成泊松分布的flow
            self.need[str(i)] = [np.random.poisson(q) for q in self.need[str(i)]]

        self.start_region = self.need.groupby('start_region')
        self.end_region = self.need.groupby('end_region')
        self.t_index = {i: str(i) for i in range(self.episode_num)}
        self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(self.episode_num)])
        self.in_nums = np.array([self.end_region[str(i)].agg(np.sum) for i in range(self.episode_num)])

    def eval_flow_init(self):   #使用原版need 固定flow 用于评价

        self.need = need.copy()
        self.start_region = self.need.groupby('start_region')
        self.end_region = self.need.groupby('end_region')
        self.t_index = {i: str(i) for i in range(self.episode_num)}
        self.out_nums = np.array([self.start_region[str(i)].agg(np.sum) for i in range(self.episode_num)])
        self.in_nums = np.array([self.end_region[str(i)].agg(np.sum) for i in range(self.episode_num)])

    def init(self):
        self.obs = self.obs_init.copy()
        self.t = 0
        return np.append(self.obs, self.t)

    def determine_init_state(self):
        region_outheat=[ sum(x) for x in zip(*self.out_nums) ]
        rate=8500/sum(region_outheat)
        self.init_state=[np.ceil(i*rate) for i in region_outheat]

    def check_limit(self,arg):  #对一个state(不含t,14位),action,t(当前)是否合法 Todo:check
        tmp_obs, action, current_car, current_eps =arg

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10

        #\ and (tmp_obs[-self.obs_dim + region] - self.out_nums[int(current_eps+1), region]) * move <= 0
        if move + tmp_obs[-self.obs_dim + region] >= 0 and move <= tmp_obs[-self.car_num+int(current_car)] :
            return False #合法动作
        else:
            return True   #非法动作

    def get_feasible_action(self,state_with_t):  #对一个state，给出合法的action组 Todo:check

        feasible_action = list()
        feasible_move=list()
        feasible_region=list()

        tmp_obs = copy.deepcopy(state_with_t[:-1])
        tmp_obs[:self.obs_dim] = tmp_obs[-self.obs_dim:]  # 更新状态

        current_car = (state_with_t[-1]) % self.car_num
        current_eps = np.floor((state_with_t[-1]) / self.car_num)
        # if current_car==0:
        #     tmp_obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(current_eps),]


        result=pool.map(self.check_limit, [(tmp_obs, action, current_car, current_eps) for action in range(self.action_dim)])
        # result=[]
        # for action in range(self.action_dim):
        #     result.append(self.check_limit((tmp_obs, action, current_car, current_eps)))

        for action,r in enumerate(result):
            if not r:
                feasible_action.append(action)
                move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10
                region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
                feasible_move.append(move)
                feasible_region.append(region)
        return feasible_action,feasible_move,feasible_region

    def calc_tmp_R(self):

        tmp_obs=self.obs.copy()
        tmp_obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(self.current_eps),]
        # tmp_obs[-self.obs_dim:-2 * self.car_num] -= self.out_nums[self.current_eps + 1,]
        raw_R = np.mean(
            [stats.poisson.cdf(i, j) for i, j in zip(tmp_obs[-self.obs_dim:-2 * self.car_num], self.out_nums[self.current_eps + 1,])])
        # raw_R=np.sum(tmp_obs[-self.obs_dim:-2 * self.car_num][tmp_obs[-self.obs_dim:-2 * self.car_num] < 0])

        return raw_R


    def step(self, action,fore_R):

        #当前决策周期+决策车辆
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

        self.obs[:self.obs_dim]=self.obs[-self.obs_dim:]  #更新状态

        region = int(np.floor(action / (2 * self.move_amount_limit + 1)))
        move = (action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)*10

        # 更新单车分布状态
        # 处理上时段骑入

        #计算初始R
        if self.current_car == 0:
            fore_R=self.calc_tmp_R()

        # 筛选不合理情况 若合理 按照推算移动车辆 更新货车状态
        # if self.check_limit(action,self.t):
        self.obs[-self.obs_dim+region] += move
        # 更新货车状态
        self.obs[-self.car_num+self.current_car] -= move  # 更新货车上的单车数
        self.obs[-2*self.car_num+self.current_car] = region  # 更新货车位置
        # 更新之前的动作历史
        self.obs[-self.obs_dim-1] = move  # 搬动的单车数
        self.obs[-self.obs_dim-2] = region  # 货车位置

        recent_R = self.calc_tmp_R()
        reward = recent_R - fore_R

        if self.current_car == self.car_num - 1:
            self.obs[-self.obs_dim:-2 * self.car_num] += self.in_nums[int(self.current_eps),]
            self.obs[-self.obs_dim:-2*self.car_num] -= self.out_nums[self.current_eps+1,]
            self.obs[-self.obs_dim:-2*self.car_num][self.obs [-self.obs_dim:-2*self.car_num]< 0] = 0

        if self.current_car == 0:  #如果是阶段起初则返回 原始R和当前R
            return np.append(self.obs, self.t), reward, fore_R, recent_R, done
        else:
            return np.append(self.obs, self.t), reward, recent_R, done


class Net(nn.Module):
    def __init__(self, NUM_STATES,car_num):
        super(Net, self).__init__()

        EMB_SIZE = 100
        OTHER_SIZE = NUM_STATES+2-2*car_num-2  #15

        self.fc1 = nn.Linear(OTHER_SIZE + EMB_SIZE * (2*car_num+2), 256).cuda()
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
    def __init__(self, NUM_STATES, NUM_ACTIONS, region_num,move_amount_limit, eps_num,car_num):
        self.eval_net, self.target_net = Net(NUM_STATES,car_num), Net(NUM_STATES,car_num)
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
        self.car_num=car_num
        self.fig, self.ax = plt.subplots()
        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.all_eps_num = (eps_num-1) * car_num
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
        state_1 = np.delete(state, [i for i in range(self.region_num, self.region_num + self.car_num)] +
                            [self.region_num + self.car_num * 2] +
                            [i for i in range(self.region_num * 2 + self.car_num * 2 + 2,
                                              self.region_num * 2 + self.car_num * 3 + 2)])

        state_2 = np.hstack((state[self.region_num:self.region_num + self.car_num],
                             [state[self.region_num + 2 * self.car_num]],
                             state[
                             2 * self.region_num + 2 * self.car_num + 2:2 * self.region_num + 3 * self.car_num + 2]))
        tmp_x=list()
        tmp_y=list()
        feasible_action,m,r=env.get_feasible_action(state)
        for move,region in zip(m,r):
            tmp_x.append(np.concatenate([state_1, np.array([move])]))
            tmp_y.append(np.concatenate([state_2, np.array([region])]))

        x = torch.FloatTensor(tmp_x).cuda()
        station = torch.LongTensor(tmp_y).cuda()

        action_val = self.target_net.forward(x, station)

        arr_action_val = action_val.detach().cpu().numpy().reshape(len(action_val))

        max_indice = [i for i, j in enumerate(arr_action_val) if
                      j == np.max(arr_action_val)]  # 找最大index
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

        x = torch.FloatTensor(np.delete(batch_memory[:, :self.NUM_STATES],
                                        [i for i in range(self.region_num,self.region_num + self.car_num)]+
                                        [self.region_num+self.car_num*2]+
                                        [i for i in range(self.region_num*2+self.car_num*2+2,self.region_num*2+self.car_num*3+2)], 1)).cuda()
                                         # -1-self.car_num-1,-1-self.car_num-2,-1-self.car_num*2-self.region_num-2,-1-self.car_num*3-self.region_num-2,-1-self.car_num*3-self.region_num-3], 1)).cuda()

        state_2 = np.hstack((batch_memory[:,self.region_num:self.region_num + self.car_num],
                             batch_memory[:,self.region_num + 2 * self.car_num:self.region_num + 2 * self.car_num+1],
                             batch_memory[:,2*self.region_num + 2 * self.car_num+2:2*self.region_num + 3 * self.car_num+2]))

        batch_reward = torch.FloatTensor(batch_memory[:, self.NUM_STATES + 1: self.NUM_STATES + 2]).cuda()

        # x=torch.FloatTensor(state_1).cuda()
        move = torch.FloatTensor([[i[0] % (2 * self.move_amount_limit + 1) - self.move_amount_limit] for i in
                                  batch_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        x = torch.cat((x, move), 1)

        y=torch.LongTensor(state_2).cuda()
        region = torch.LongTensor([[int(np.floor(i[0] / (2 * self.move_amount_limit + 1)))] for i in
                                   batch_memory[:, self.NUM_STATES:self.NUM_STATES + 1]]).cuda()
        y = torch.cat((y, region), 1)

        q_eval = self.eval_net(x, y)

        tmp_q_next = list()
        for state in batch_memory[:, -self.NUM_STATES:]:
            if state[-1]!=self.all_eps_num:
                feasible_action, m, r = env.get_feasible_action(state)

                tmp_x = list()
                tmp_y = list()
                # 对每个feasible action算value
                state_1 = np.delete(state,[i for i in range(self.region_num, self.region_num + self.car_num)] +
                          [self.region_num + self.car_num * 2] +
                          [i for i in range(self.region_num * 2 + self.car_num * 2 + 2,
                                            self.region_num * 2 + self.car_num * 3 + 2)])

                state_2 = np.hstack((state[self.region_num:self.region_num + self.car_num],
                                     [state[self.region_num + 2 * self.car_num]],
                                     state[2 * self.region_num + 2 * self.car_num + 2:2 * self.region_num + 3 * self.car_num + 2]))

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
            env.eval_flow_init()
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
                    f"move:{10*(action % (2 * self.move_amount_limit + 1) - self.move_amount_limit)} reward:{reward} "
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
    car_num=2
    EPSILON = 0.99
    EPS_DECAY = 0.999
    env = Env(region_num=58, move_amount_limit=50, eps_num=eps_num,car_num=car_num)
    NUM_ACTIONS = (2 * env.move_amount_limit + 1) * env.region_num  # [-500,500]*4个方块
    NUM_STATES = 2*env.region_num + 4*car_num+ 2 + 1 #19


    net = Dqn(NUM_STATES, NUM_ACTIONS, env.region_num, env.move_amount_limit, eps_num,car_num)
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in range(EPISODES):
        if episode % 2000 ==0:
            env.possion_flow_init()
            print(env.out_nums)
        state = env.init()
        step_counter = 0
        reward_sum = 0
        history_action=[]
        EPSILON = max(EPSILON * EPS_DECAY, 0.01)
        fore_R=0
        while True:
            step_counter += 1
            # env.render()
            action = net.choose_action(state,EPSILON,env)

            move = (action % (2 * env.move_amount_limit + 1) - env.move_amount_limit)*10
            region=int(np.floor(action / (2 * env.move_amount_limit + 1)))
            history_action.append((region,move))

            if env.t%env.car_num==0:
                next_state, reward, raw_R, fore_R, done = env.step(action, fore_R)
                reward_sum+=raw_R
                net.store_trans(state, action, reward+raw_R, next_state)

            else:
                next_state, reward, fore_R, done = env.step(action, fore_R)
                net.store_trans(state, action, reward, next_state)

            # print(next_state,reward)
            # net.store_trans(state, action, reward, next_state)
            reward_sum += reward

            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn(env)
                # if done:
                #     print("episode {}, the reward is {}".format(episode, round(reward_sum, 3)))
                    # print(f"{round(reward_sum, 3)}", file=open(f"result_history/actionless_output_result_{ts}.txt", "a"))

            if done:
                print("episode {}, the reward is {}, history action {}".format(episode, round(reward_sum/(eps_num-1), 3),history_action))
                print(f"{round(reward_sum, 3)}", file=open(f"result_history/real_output_result_move_amount_limit50_{ts}.txt", "a"))
                break

            state = next_state

        if episode % 100 == 0:
            te = time.time()
            print(f'time consume: {te - ts}')
    print(net.evaluate(env))


if __name__ == '__main__':
    main()
