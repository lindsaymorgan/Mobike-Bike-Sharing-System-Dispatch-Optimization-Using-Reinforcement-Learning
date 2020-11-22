import numpy as np
from copy import deepcopy
import scipy.stats as stats

class State:
    def __init__(self, state, region_count, car_num, out_nums, in_nums, capacity_each_step, reward=0, t=0, reward_sum=0, R=None):
        self.region_count = region_count
        self.car_num = car_num
        self.state = state
        self.out_nums = out_nums
        self.in_nums = in_nums
        self.capacity_each_step = capacity_each_step
        self.reward = reward
        self.reward_sum = reward_sum
        self._R = R
        self.t = t
        self.__hash = None
        self.feasible_actions = np.zeros((self.region_count, 2 * self.capacity_each_step + 1))

    def get_hash(self):
        if not self.__hash:
            self.__hash = tuple(self.state).__hash__()
        return self.__hash

    def __repr__(self):
        return str(tuple(self.state))

    @property
    def region_state(self):
        return self.state[:self.region_count]

    @region_state.setter
    def region_state(self, value):
        self._R = None
        self.__hash = None
        self.state[:self.region_count] = value

    @property
    def car_pos(self):
        return self.state[self.region_count:self.region_count + self.car_num]

    @car_pos.setter
    def car_pos(self, value):
        self.state[self.region_count:self.region_count + self.car_num] = value

    @property
    def bike_on_car(self):
        return self.state[self.region_count + self.car_num:]

    @bike_on_car.setter
    def bike_on_car(self, value):
        self.state[self.region_count + self.car_num:] = value

    @property
    def R(self) -> int:
        """
        :return: Reward
        """
        # if self._R:
        #     return self._R
        # self.region_state += self.in_nums[self.t,]
        # self.region_state -= self.out_nums[self.t+1 ,]
        # raw_R = np.sum(self.region_state[self.region_state < 0])
        # self.region_state += self.out_nums[self.t+1 ,]
        # self.region_state -= self.in_nums[self.t]

        self.region_state += self.in_nums[self.t,]
        raw_R = np.mean(
            [stats.poisson.cdf(i, j) for i, j in zip(self.region_state, self.out_nums[self.t + 1,])])
        self.region_state -= self.in_nums[self.t]
        self._R = raw_R
        return raw_R

    def out_stage(self):
        """
        before move happens -- external bikes depart
        """
        self.region_state -= self.out_nums[self.t,]
        self.region_state[self.region_state < 0] = 0
        return self.region_state

    def in_stage(self):
        """
        after move happens -- external bikes arrive
        """

        self.region_state += self.in_nums[self.t,]
        self.t += 1

    def check_feasible(self, current_region, current_car, move) -> bool:
        """
        Return True for feasible action, False for not feasible
        :param state: State object, state to check
        :param current_region: index of region
        :param move: number of bikes to load/unload (must be within -capacity_each_step ~ capacity_each_step)
        :param current_car: index of car
        :return:
        """

        # \ and (tmp_obs[-self.obs_dim + region] - self.out_nums[int(current_eps+1), region]) * move <= 0
        #move 正数移入区块 负数移出区块
        if move + self.region_state[current_region] >= 0 and move <= self.bike_on_car[current_car]:
            return True  # 合法动作
        else:
            return False  # 非法动作

    def update_feasible_action(self, current_car):
        for region in range(self.region_count):
            for move in range(-self.capacity_each_step, self.capacity_each_step + 1):
                self.feasible_actions[region, move] = self.check_feasible(region, current_car, move)


    def step(self, current_region, current_car, move, prev_state_R=None):
        """
        Perform move action
        :param current_region:
        :param current_car:
        :param move:
        :param prev_state_R:
        :return:
        """
        new_state = State(deepcopy(self.state), self.region_count,
                          self.car_num, self.out_nums, self.in_nums,
                          self.reward, self.t, self.reward_sum, self.R)
        # if (move > 0 or move + new_state.region_state[current_region] >= 0) and move <= new_state.bike_on_car[current_car]:
        if move + new_state.region_state[current_region] >= 0 and move <= new_state.bike_on_car[current_car]:
            new_state.region_state[current_region] += move
            # 更新货车状态
            new_state.bike_on_car[current_car] -= move  # 更新货车上的单车数
            new_state.car_pos[current_car] = current_region  # 更新货车位置
        new_state.reward = new_state.R
        if prev_state_R:
            new_state.reward -= prev_state_R
        new_state.reward_sum += new_state.reward
        return new_state


class Env(object):
    def __init__(self, initial_region_state, capacity_each_step, max_episode, car_count, need):
        """
        :param initial_region_state: List, number of bikes in each region, e.g. [15, 15, 15, 15]
        :param capacity_each_step: maximum number of load/unload bikes each step (only one of load/unload per step)
        :param max_episode: max time
        :param car_count: number of cars
        :param need: external change driven by customers
        """
        self.initial_region_state = initial_region_state
        self.region_count = len(initial_region_state)
        self.capacity_each_step = capacity_each_step
        self.car_num = car_count

        # length of one-hot action vector: for each region, each car can load/unload maximum transport_capacity of bike
        self.action_dim = self.region_count * (2 * self.capacity_each_step + 1)

        # length of state: number of bike at each region + location of each car + number of bike on each car
        self.obs_dim = self.region_count + 2 * self.car_num

        self.start_region = need.groupby('start_region')
        self.end_region = need.groupby('end_region')
        self.t_index = {i: str(i) for i in range(max_episode + 1)}
        self.out_nums = np.array([need.groupby('start_region')[str(i)].agg(np.sum) for i in range(max_episode + 1)])
        self.in_nums = np.array([need.groupby('end_region')[str(i)].agg(np.sum) for i in range(max_episode + 1)])

        # current episode
        self.t = 0

    def new_state(self):
        """
        Initialize state
        :return:
        """
        state = State(np.asarray(self.initial_region_state + [0] * self.car_num * 2), self.region_count,
                      self.car_num, self.out_nums, self.in_nums, self.capacity_each_step)
        return state
