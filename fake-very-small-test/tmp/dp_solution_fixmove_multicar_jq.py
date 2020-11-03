import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import scipy.stats as stats
from tqdm import tqdm

from collections import namedtuple
StateVars = namedtuple('state_vars', ['curr_state', 'prev_state_hash', 'reward'])
from Environment import State, Env

need = pd.read_csv('../fake_4region_trip_20170510.csv')
# dist=pd.read_csv('fake_4region_distance.csv')
# dist=dist.values
eps_num = 4
car_num = 1

env = Env(initial_region_state=[15, 15, 15, 15], capacity_each_step=10, max_episode=eps_num, car_count=car_num, need=need)
history = {i: dict() for i in range(8)}


for region in range(env.region_count):
    state = env.new_state()
    curr_state_hash = state.get_hash()
    state.out_stage()
    for car in range(env.car_num):
        for move in range(-env.capacity_each_step, env.capacity_each_step + 1):
            if state.check_feasible(region, car, move):

                new_state = state.step(region, car, move)
                new_state.in_stage()
                new_state_hash = new_state.get_hash()
                # new_state.out_stage()
                if (new_state_hash not in history[0] or
                        new_state_hash in history[0] and history[0][new_state_hash].reward < new_state.reward):
                    history[0][new_state_hash] = StateVars(curr_state=new_state, prev_state_hash=curr_state_hash, reward=new_state.reward)
    state.in_stage()
for stage in tqdm(range(1, eps_num), desc='stage'):
    for prev_state_hash, state_vars in history[stage - 1].items():
        curr_state = state_vars.curr_state
        curr_state.out_stage()
        for region in range(env.region_count):
            for car in range(env.car_num):
                for move in range(-env.capacity_each_step, env.capacity_each_step + 1):
                    if curr_state.check_feasible(current_region=region, current_car=car, move=move):
                        new_state = curr_state.step(region, car, move, curr_state.R)
                        new_state_hash = new_state.get_hash()
                        if stage < eps_num - 1:
                            new_state.in_stage()
                            # new_state.out_stage()
                        if (new_state_hash not in history[stage] or
                                new_state_hash in history[stage] and history[stage][new_state_hash].reward < new_state.reward):
                            history[stage][new_state_hash] = StateVars(curr_state=new_state, prev_state_hash=prev_state_hash, reward=new_state.reward)
        curr_state.in_stage()
max_reward = max([i.reward for i in history[eps_num - 1].values()])
print([i.reward for i in history[eps_num - 1].values()])
print(max_reward)
new_state_hash = [i for i, v in history[eps_num - 1].items() if v.reward == max_reward][0]

ts = int(time.time())
# outfile=f"result_action/fakesmall_2cars_output_action_{ts}.txt"
reward_sum = 0
for i in range(eps_num - 1, -1, -1):
    # print(max_state)
    # print(i)
    print(i, history[i][new_state_hash])
    # print(max_state,history_action[i][max_state],
    #                     file=open(outfile, "a"))
    reward_sum += history[i][new_state_hash].reward
    if i != 0:
        new_state_hash = history[i][new_state_hash].prev_state_hash
        # print(max_state)

print(f"best reward : {round(reward_sum / (eps_num - 1), 3)}")
# print(f"best reward : {round(reward_sum/(eps_num-1), 3)}",
#                     file=open(outfile, "a"))
