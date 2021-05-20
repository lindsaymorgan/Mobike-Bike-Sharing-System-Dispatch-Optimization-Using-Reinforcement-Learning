import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
import pandas as pd
from tqdm import tqdm
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt

need=pd.read_csv('4region_trip_20170510_eachhour.csv')

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
######################################################################
######################################################################
#
# 1. 请设定 learning rate，可以从 0.001 起调，尝试增减
#
######################################################################
######################################################################
LEARNING_RATE = 0.001 # 学习率

class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 256
        hid2_size = 256
        hid3_size=256
        # 3层全连接网络
        self.emb_1 = layers.embedding(size=[128, 64])
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc4 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        cast = fluid.layers.cast(x=obs, dtype='int64')
        fluid.layers.Print(cast)
        # out = fluid.layers.sequence_expand(x=cast, ref_level=0)
        # results = exe.run(fluid.default_main_program(),
        #                   feed={'x': cast},
        #                   fetch_list=[out], return_numpy=False)
        h1 = self.emb_1(cast)
        h2 = self.fc1(h1)
        h3 = self.fc2(h2)

        Q = self.fc4(h3)

        return Q

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
                self.obs[i] = 0
            self.obs[self.region_num + region] = 1  # 更新货车位置


        for i in range(self.region_num):

            if self.obs[i] > out_num[i]:
                self.obs[i]-=out_num[i]

            #如果不能满足时间段内的所有需求
            else:
                reward+=(self.obs[i]-out_num[i]) #不能满足的部分设为损失
                self.obs[i]=0  #设余量为0

        return self.obs, reward, done, self.t



from parl.algorithms import DQN

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 move_amount_limit,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.move_amount_limit =move_amount_limit
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
            # region = int(np.floor(act / (2 * self.move_amount_limit + 1)))
            # move = act % (2 * self.move_amount_limit + 1) - self.move_amount_limit
            # # 判断动作是否合法（搬出量不超过region单车量 & 搬入量不超过车上的单车总量）
            # print(obs[region],obs[-1],move)
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        flag = 0
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)

        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        # region = int(np.floor(act / (2 * self.move_amount_limit + 1)))
        # move = act % (2 * self.move_amount_limit + 1) - self.move_amount_limit
        # # 判断动作是否合法（搬出量不超过region单车量 & 搬入量不超过车上的单车总量）
        # print(obs[region], obs[-1], move)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

# replay_memory.py
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, t_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, t = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            t_batch.append(t)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(t_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到

        next_obs, reward, done, t = env.step(action)

        rpm.append((obs, action, reward, next_obs, t))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_t) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_t)  # s,a,r,s',t

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, t = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# 创建环境
# os.environ["CUDA_VISIBLE_DEVICES"] = ''
# env = gym.make('MountainCar-v0')
# action_dim = env.action_space.n  # MountainCar-v0: 3
# obs_shape = env.observation_space.shape  # MountainCar-v0: (2,)

env=Env(region_num=4,move_amount_limit=500,eps_num=23)
action_dim = (2*env.move_amount_limit+1)*env.region_num  # [-500,500]*4个方块
obs_shape = (2*env.region_num+1,0 ) # MountainCar-v0: (2,)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池



# 根据parl框架构建agent
######################################################################
######################################################################
#
# 4. 请参考课堂Demo，嵌套Model, DQN, Agent构建 agent
#
######################################################################
######################################################################
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    move_amount_limit=500,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低



# 加载模型
# save_path = './dqn_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 2000

# 开始训练
episode = 0
tr=list()
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in tqdm(range(50)):
        total_reward = run_episode(env, agent, rpm)
        episode += 1

    # test part
    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))
    tr.append(eval_reward)
    # if episode%10==0:
    #     plt.plot([i for i in range(len(tr))],tr,'o-')
    #     plt.show()

# 训练结束，保存模型
save_path = './dqn_model.ckpt'
agent.save(save_path)
print(evaluate,evaluate(env, agent, render=False))