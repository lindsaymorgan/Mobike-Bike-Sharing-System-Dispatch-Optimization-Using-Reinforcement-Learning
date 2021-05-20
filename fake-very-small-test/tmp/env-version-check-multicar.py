from Environment import Env
import numpy as np
import pandas as pd

initial_region_state=[15,15,15,15]
capacity_each_step=10
max_episode=5
car_count=1
need=pd.read_csv('../fake_4region_trip_20170510.csv')

env = Env(initial_region_state, capacity_each_step, max_episode, car_count, need)
NUM_ACTIONS = (2 * env.capacity_each_step + 1) * env.region_count  # [-500,500]*4个方块
NUM_STATES = 2 * env.region_count + 7  # MountainCar-v0: (2,)

history_dict={0: dict(),1: dict(),2: dict(),3:dict(),4:dict(),5: dict(),6: dict(),7: dict()}
history_action={0: dict(),1: dict(),2: dict(),3:dict(),4:dict(),5: dict(),6: dict(),7: dict()}

state = env.init()
print(state)

for action in range(NUM_ACTIONS):
    env.reset()
    env.pre_step()
    move = action % (2 * env.capacity_each_step + 1) - env.capacity_each_step
    region = int(np.floor(action / (2 * env.capacity_each_step + 1)))
    if env.check_feasible(env.state,region,0,move):
        state,reward, recent_R=env.step(region,0,move)
        if (state in history_dict[0] and history_dict[0][state] < reward) \
                or state not in history_dict[0]:
            history_dict[0][state] = (reward,recent_R)  #记录 state->reward R
            history_action[0][state] =(move,region,reward) #记录 state->move region reward

#逐个阶段
for stage in range(1,(max_episode-1)*car_count):

    current_car = stage % car_count
    current_eps = int(np.floor(stage / car_count))

    for action in range(NUM_ACTIONS):

        move = action % (2 * env.capacity_each_step + 1) - env.capacity_each_step
        region = int(np.floor(action / (2 * env.capacity_each_step + 1)))

        for state_fore,(reward_fore,fore_R) in history_dict[stage-1].items():


            if not env.check_feasible(np.asarray(state_fore), region, current_car, move):
                state, reward, fore_R = env.step(region,current_car,move)
                if (state in history_dict[stage] and history_dict[stage][state][0] <reward+reward_fore)\
                        or state not in history_dict[stage]:
                    history_dict[stage][state] = (reward+reward_fore,fore_R)
                    history_action[stage][state] = (move,region,state_fore,reward)

max_value=max([i[0] for i in history_dict[(eps_num-1)*car_num-1].values()])
print(max_value)
max_state=[i for i,v in history_dict[3].items() if v[0]==max_value][0]

ts=int(time.time())
# outfile=f"result_action/fakesmall_2cars_output_action_{ts}.txt"
reward_sum=0
for i in reversed(range((eps_num-1)*car_num)):
    # print(max_state)
    # print(i)
    print(max_state, history_action[i][max_state])
    # print(max_state,history_action[i][max_state],
    #                     file=open(outfile, "a"))
    reward_sum+=history_action[i][max_state][-1]
    if i!=0:
        max_state=history_action[i][max_state][2]
        # print(max_state)

print(f"best reward : {round(reward_sum/(eps_num-1), 3)}")
# print(f"best reward : {round(reward_sum/(eps_num-1), 3)}",
#                     file=open(outfile, "a"))
