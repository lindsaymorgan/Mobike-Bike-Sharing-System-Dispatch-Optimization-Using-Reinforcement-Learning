##########################################################
# AUTHOR:            Ai Kagawa
# DATE CREATED:      11/23/2018
# DATA LAST UPDATED: 03/08/2019

# DESCRIPTION:
# This code implements Deep Q-Learning (DQN) for the mountain car problem
# (https://github.com/openai/gym/wiki/MountainCar-v0).
##########################################################

import time
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

############################# command line arguments #############################

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-figure', help='Do you want to print figures?', required=False)
parser.add_argument('-print', help='Do you want to print details?', required=False)
parser.add_argument('-episodes', type=int, help='Number of Episodes', required=False)
parser.add_argument('-steps', type=int, help='Number of Time Steps', required=False)
parser.add_argument('-alpha', type=float, help='Learning Rate', required=False)
parser.add_argument('-gamma', type=float, help='Discount Factor', required=False)
parser.add_argument('-batch_size', type=int, help='Batch Size', required=False)
parser.add_argument('-target_update', type=int, help='Target Update Frequency', required=False)
parser.add_argument('-memory_capacity', type=int, help='Replay Memory Capacity', required=False)
parser.add_argument('-eps', type=float, help='Epsilon Greedy', required=False)
parser.add_argument('-eps_min', type=float, help='Minimum Epsilon Greedy', required=False)
parser.add_argument('-eps_decay', type=float, help='Decay Rate of Epsilon Greedy', required=False)

args = vars(parser.parse_args())

############################# Parameters (with default values) #############################

showFigures     = False if args['figure'] == 'false' else True # show figures
printDetails    = False if args['print'] == 'false' else True  # print details
EPISODES        = 100  if args['episodes'] is None else args['episodes'] # # of episodes to train
TIMESTEPS       = 200  if args['steps'] is None else args['steps'] # # of time steps in each episode
ALPHA           = 0.01 if args['alpha'] == None else args['alpha'] # learning rate
GAMMA           = 0.9  if args['gamma'] == None else args['gamma'] # discount factor
BATCH_SIZE      = 32   if args['batch_size'] == None else args['batch_size'] # batch size
TARGET_UPDATE   = 100  if args['target_update'] == None else args['target_update'] # target update frequency
MEMORY_CAPACITY = 2000 if args['memory_capacity'] == None else args['memory_capacity'] # replay memory capacity
EPS             = 0.9  if args['eps'] is None else args['eps'] # epsilon greedy policy
EPS_MIN         = 0.01 if args['eps_min'] is None else args['eps_min'] # minimum epsilon
EPS_DECAY       = 0.99 if args['eps_decay'] is None else args['eps_decay'] # decay rate of epsilon

############################# Setup the problem #############################

env = gym.make('MountainCar-v0').unwrapped
STATE_SIZE  = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

# if gpu is to be used
device = "cpu"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################# Neural Net class #############################

class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 32)
        self.fc2 = nn.Linear(32, ACTION_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

############################# DQN algorithm class #############################

class DQN(object):

    def __init__(self):

        self.policy_net    = Net().to(device) # policy network
        self.target_net    = Net().to(device) # target network
        self.target_net.load_state_dict(self.policy_net.state_dict())  # copy the policy net parameters to target net
        self.learn_counter = 0 # for target updating
        self.mem_counter   = 0 # for counting memory
        self.memory        = np.zeros((MEMORY_CAPACITY, STATE_SIZE * 2 + 2))  # initialize replay memory
        self.optimizer     = torch.optim.Adam(self.policy_net.parameters(), lr=ALPHA)  # optimizer
        self.loss_func     = nn.MSELoss() # loss function

        self.success_counter = 0  # counting # of consecutive successes
        self.isSuccess = False    # is successed
        self.isStartedLearning = True


    def select_action(self, state):

        rand_num = random.random()
        global EPS
        if EPS * EPS_DECAY >= EPS_MIN : EPS = EPS * EPS_DECAY

        if rand_num > EPS: # greedy
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            Q = self.policy_net.forward(state)
            return torch.max(Q, 1)[1].data.numpy()[0]
            #with torch.no_grad():
            #    return self.policy_net(state).max(1)[1].view(1, 1)
        else: # random
            return np.random.randint(0,ACTION_SIZE)


    def store_transition(self, s, a, r, s1):
        transition = np.hstack((s, [a, r], s1))
        index = self.mem_counter % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.mem_counter += 1


    def learn(self):

        # target parameter update
        if self.learn_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory     = self.memory[sample_index, :]

        batch_state  = torch.FloatTensor(batch_memory[:, :STATE_SIZE])
        b_a  = torch.LongTensor(batch_memory[:, STATE_SIZE:STATE_SIZE+1].astype(int))
        b_r  = torch.FloatTensor(batch_memory[:, STATE_SIZE+1:STATE_SIZE+2])
        batch_state1 = torch.FloatTensor(batch_memory[:, -STATE_SIZE:])

        # q_eval w.r.t the action in experience
        Q      = self.policy_net(batch_state).gather(1, b_a)  # shape (batch, 1)
        Q1     = self.target_net(batch_state1).detach()     # detach from graph, don't backpropagate
        target = b_r + GAMMA * Q1.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss   = self.loss_func(Q, target)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self):

        if printDetails : print('\n################### Start collecting experience ###################')

        for e in range(EPISODES):

            s = env.reset()
            score = 0

            while True:

                if showFigures: env.render()
                a = self.select_action(s)        # select action
                s1, r, done, info = env.step(a)  # take action
                score -= r  # accumulate the original score
                r = self.reward_shaping(s1)
                self.store_transition(s, a, r, s1)  # store transition in the replay memory

                # if enough trajectories in the replay memory, start training DQN
                if self.mem_counter > 5*BATCH_SIZE:  # I randomly picked this size! :)
                    if self.isStartedLearning : self.start_learning_comment()
                    self.learn()

                if done or score>=TIMESTEPS:
                    if printDetails : print('Episode: ', e, '\tReward: ', int(-score))
                    self.success_tracker(score)
                    break

                s = s1

            if self.success_counter >= 5:
                self.print_result(e)
                break


    def reward_shaping(self, s1):   # imprve reward function
        position, velocity = s1
        return abs( position +.5 )


    def success_tracker(self, score):
        if score < TIMESTEPS:
            if not self.isSuccess : self.success_counter = 0
            self.isSuccess = True
            self.success_counter += 1
        else:
            self.isSuccess = False


    def start_learning_comment(self):
        if printDetails : print('\n################### Start learning ###################')
        self.isStartedLearning = False


    def print_result(self, e):
        if printDetails : print("Finished training")
        print("DQN learned in ", e, " trials!")

############################# main function #############################

def main():

    start = time.time()
    dqn = DQN()
    dqn.train()
    end = time.time()
    print("Time: ", end - start)

    env.close()

main()