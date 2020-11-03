import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import scipy.stats as stats
from tqdm import tqdm

from collections import namedtuple
StateVars = namedtuple('state_vars', ['curr_state', 'prev_state_hash', 'reward','move','region'])
from Environment import State, Env

need = pd.read_csv('fake_4region_trip_20170510.csv')
# dist=pd.read_csv('fake_4region_distance.csv')
# dist=dist.values
eps_num = 4
car_num = 2

env = Env(initial_region_state=[15, 15, 15, 15], capacity_each_step=3, max_episode=eps_num, car_count=car_num, need=need)
history = {i: dict() for i in range(eps_num*car_num)}


# for car in range(env.car_num):
car=0
small_eps=0
for region in range(env.region_count):
    for move in range(-env.capacity_each_step, env.capacity_each_step + 1):
        state = env.new_state()
        curr_state_hash = state.get_hash()
        state.out_stage()
        # state.in_stage()
        if state.check_feasible(region, car, move):

            new_state = state.step(region, car, move)

            if car == env.car_num - 1:
                new_state.in_stage()
                new_state.out_stage()

            new_state_hash = new_state.get_hash()
            # new_state.out_stage()
            if (new_state_hash not in history[0] or
                    new_state_hash in history[0] and history[0][new_state_hash].reward < new_state.reward_sum):
                history[0][new_state_hash] = StateVars(curr_state=new_state, prev_state_hash=curr_state_hash, reward=new_state.reward, move=move,region=region)
# state.in_stage()

for stage in tqdm(range(eps_num), desc='stage'):   # Todo: check t

    for car in range(env.car_num):
        if stage==0 and car==0:
            continue
        for prev_state_hash, state_vars in history[stage*env.car_num+car - 1].items():
            curr_state = state_vars.curr_state
            for region in range(env.region_count):
                for move in range(-env.capacity_each_step, env.capacity_each_step + 1):
                    if curr_state.check_feasible(current_region=region, current_car=car, move=move):
                        if car==0:
                            new_state = curr_state.step(region, car, move)
                        else:
                            new_state = curr_state.step(region, car, move, curr_state.R)

                        if car==env.car_num-1:
                            new_state.in_stage()
                            new_state.out_stage()

                        new_state_hash = new_state.get_hash()
                        if (new_state_hash not in history[stage*env.car_num+car] or
                                (new_state_hash in history[stage*env.car_num+car] and history[stage*env.car_num+car][new_state_hash].reward < new_state.reward_sum)):
                            history[stage*env.car_num+car][new_state_hash] = StateVars(curr_state=new_state, prev_state_hash=prev_state_hash, move=move,region=region,reward=new_state.reward_sum)

ts = int(time.time())
outfile=f"result_action/fakesmall_2cars_output_reliab_capacity_each_step_3_{ts}.txt"

max_reward = max([i.reward for i in history[eps_num*car_num - 1].values()])
print(max_reward)
new_state_hash = [i for i, v in history[eps_num*car_num - 1].items() if v.reward == max_reward][0]

reward_sum = 0
for i in range(eps_num*car_num - 1, -1, -1):

    print(i, history[i][new_state_hash],file=open(outfile, "a"))
    print(i, history[i][new_state_hash])

    reward_sum += history[i][new_state_hash].reward
    if i != 0:
        new_state_hash = history[i][new_state_hash].prev_state_hash
        # print(max_state)

print(f"best reward : {round(reward_sum / (eps_num), 3)}")
print(f"best reward : {round(reward_sum/(eps_num), 3)}",
                    file=open(outfile, "a"))
