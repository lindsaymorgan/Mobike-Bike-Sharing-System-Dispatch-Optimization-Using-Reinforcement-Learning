import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import gym
import random

#hyper parameters

GAMMA = 0.9
LR = 0.01
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32

EPISODES = 4000
env = gym.make('MountainCar-v0')
# env._max_episode_steps = 4000
# env = env.unwrapped
NUM_STATES = env.observation_space.shape[0] # 2
NUM_ACTIONS = env.action_space.n


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(NUM_STATES,32)
        # self.fc1.weight.data.normal_(0, 0.1)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(32, NUM_ACTIONS)
        # self.fc3.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)

        return x

class Dqn():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES *2 +2))
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.loss = nn.MSELoss()

        self.fig, self.ax = plt.subplots()

    def store_trans(self, s, a, r, s1):
        transition = np.hstack((s, [a, r], s1))
        index = self.memory_counter % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory_counter += 1

    # def store_trans(self, state, action, reward, next_state):
    #     if self.memory_counter % 1000 ==0:
    #         print("The experience pool collects {} time experience".format(self.memory_counter))
    #     index = self.memory_counter % MEMORY_CAPACITY
    #     trans = np.hstack((state, [action, reward], next_state))
    #     self.memory[index,:] = trans
    #     self.memory_counter += 1

    def choose_action(self, state,EPSILON):
        # notation that the function return the action's index nor the real action
        # EPSILON

        if random.random()> EPSILON:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()[0] # get action whose q is max
        else:
            action = np.random.randint(0,NUM_ACTIONS)

        return action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def reward_shaping(self, s1):   # imprve reward function
        position, velocity = s1
        return abs( position +.5 )

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter+=1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        #note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1: NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



def main():
    EPSILON = 0.9
    EPS_DECAY = 0.99
    e_greed_decrement = 1e-6
    net = Dqn()
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in range(EPISODES):
        state = env.reset()
        step_counter = 0
        while True:
            step_counter +=1
            env.render()

            EPSILON = max(EPSILON * EPS_DECAY, 0.01)
            action = net.choose_action(state,EPSILON)
            # EPSILON= max(EPSILON-e_greed_decrement,0.01)
            next_state, reward, done, info = env.step(action)
            reward=net.reward_shaping(next_state)
            # reward = reward * 100 if reward >0 else reward * 5
            net.store_trans(state, action, reward, next_state)
            #MEMORY_CAPACITY

            if net.memory_counter > 5*BATCH_SIZE :
                net.learn()

            if done:
                print("episode {}, the reward is {}".format(episode, step_counter))
                step_counter_list.append(step_counter)
                # print(step_counter)
                # with open('dqn-gym-step-counter.txt', 'a+') as f:
                #     f.write(f'{step_counter}\n')
                # net.plot(net.ax, step_counter_list)
                break

            state = next_state

if __name__ == '__main__':
    main()