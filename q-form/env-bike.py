class Env(object):
    def __init__(self, region_num,move_amount_limit,eps_num):
        self.region_num=region_num
        self.move_amount_limit=move_amount_limit
        self.action_dim=region_num*(2*move_amount_limit+1)
        self.obs_dim=2*region_num+1
        self.t=0
        self.epsiode_num=eps_num
        self.obs=np.array([2100,2100]) #各方格单车量+货车位置+货车上的单车量
        out_num=np.array(need.groupby('start_region')[f'{self.t}'].agg(np.sum))
        for i in range(self.region_num):
            self.obs[i]-=out_num[i]


    def reset(self):
        self.obs = np.array([2100,2100])
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
                if self.obs[i] == 1:
                    self.obs[i] = 0
                    break
            self.obs[self.region_num + region] = 1  # 更新货车位置


        for i in range(self.region_num):

            if self.obs[i] >= out_num[i]:
                self.obs[i]-=out_num[i]

            #如果不能满足时间段内的所有需求
            else:
                reward+=(self.obs[i]-out_num[i]) #不能满足的部分设为损失
                self.obs[i]=0  #设余量为0

        return self.obs, reward, done, self.t