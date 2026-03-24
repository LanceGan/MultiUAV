from entity import UAV, User
import numpy as np
import os
import sys
import math
import copy
# import Radio_env_BS as rad_env
# import radio_map as rad_env
import radio_map_A2G as A2G#A2G
import radio_map_G2A as G2A #
from numpy import linalg as LA
from rural_world import Rural_world


class World(object):
    def __init__(self, length=10, width=10, uav_num=1, user_num=10,dist_max=0.10,
                 delta_t=0.5,t=200, uav_h=1,
                 data_size = 250,ini_loc=[10.93,4.4],end_loc=[15.5,18.8],
                 users_name='Users.txt'
                 ,BS_loc=[],
                 traverse_sequence=[]):
        self.length = length # 区域大小
        self.width = width
        self.uav_num = uav_num  # 总的无人机数目
        self.users_path = users_name # 用户位置存储路径
        self.user_num = user_num  # 用户数目
        self.Users = []
        self.UAVs = []
        self.T = t # 时间步
        self.t = 0
        self.max_x = length # 区域边界
        self.min_x = 0
        self.max_y = width
        self.min_y = 0
        self.uav_h = uav_h  # 无人机飞行高度
        self.dist_max = dist_max # max flying distance m
        self.delta_T = delta_t # 时隙长度
        self.fa = 0.0  # 出界
        self.r = 0.0  # 奖励
        self.terminal = False
        self.total_engy = 0.0
        self.out_time = 0.0
        self.NON_COMM_PENALTY = -5 # 不满足通信阈值的惩罚
        self.NON_COVER_PENALTY = -10 # 不满足覆盖阈值的惩罚
        
        self.Engy_w = 0.05

        self.initial_loc = ini_loc # 起点
        self.end_loc = end_loc # 终点
        self.distance = 0.5 # m
        self.SIR_THRESHOLD_COMM = 3 # dB 通信阈值
        self.SIR_THRESHOLD_COVER = 2 # dB 通信阈值
        self.data_size_ini = data_size # 每个巡检点的数据量
        self.data_size = self.data_size_ini  # 每个巡检点的数据量
        self.data_rate = 0
        self.BandWidth = 1
        self.transmit = False # 是否完成数据传输
        self.trans_delay_start = 0
        self.trans_delay_end = self.T
        self.trans_delay = 0
        
        self.Traverse = traverse_sequence # 巡检点序列
        self.Traverse_order = 0 # 巡检点序列索引
        
        # =============Define the GT Distribution=======================
        
        self.GT_loc = BS_loc # 用户位置
        self.set_users()
        self.target = self.Users[self.Traverse[self.Traverse_order]-1]
        self.target_loc = np.array([self.target.x, self.target.y])  # 目标位置
        self.target_dis = LA.norm(self.initial_loc - self.target_loc) # 距离目标的距离
        self.target_pre_loc = np.array([self.initial_loc[0],self.initial_loc[1]]) #上一个目标的位置，初始为起点
        
        
        self.urban_world = Rural_world(self.GT_loc)
        self.HeightMapMatrix = self.urban_world.Buliding_construct()
        

    def set_users(self):
        self.Users =[]
        if os.path.exists(self.users_path): # 读写文件 # 读入用户位置
            f = open(self.users_path, 'r')
            if f:
                user_loc = f.readline()
                user_loc = user_loc.split(' ')
                self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                # self.GT_loc[len(self.Users) - 1] = np.array(
                #     [float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                while user_loc:
                    user_loc = f.readline()
                    if user_loc:
                        user_loc = user_loc.split(' ')
                        self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                        # self.GT_loc[len(self.Users) - 1] = np.array([float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                f.close()
        else:
           assert False, "Users file not found: " + self.users_path

    def reset_target(self):
        self.Traverse_order = 0
        self.target = self.Users[self.Traverse[self.Traverse_order]-1]
        self.target_loc = np.array([self.target.x, self.target.y])  # 目标位置
        self.target_dis = LA.norm(self.initial_loc - self.target_loc) # 距离目标的距离
        self.target_pre_loc = np.array([self.initial_loc[0],self.initial_loc[1]]) #上一个目标的位置，初始为起点
        
    # 重置位置，各种时间清0
    def reset(self):
        self.reset_target()
        self.set_uavs_loc()
        self.trans_delay_start = 0
        self.trans_delay_end = self.T
        self.trans_delay = 0
        self.data_size = 0
        self.transmit = True  
        self.t = 0
        self.fa = 0
        self.out_time = 0
        self.total_engy = 0.0
        state = self.reset_state()
        return state,self.t

    # def reset_loc(self):
    #     self.set_uavs_loc()
    #     state = self.reset_state()
    #     self.t = 0
    #     self.fa = 0
    #     self.out_time = 0
    #     self.total_engy = 0.0
    #     return state,self.t

    def set_uavs_loc(self):
        self.UAVs = []
        for i in range(self.uav_num):
            x = self.initial_loc[0]
            y = self.initial_loc[1]
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))
    
    def reset_state(self):
        State_dim = self.uav_num*2+2+1
        s = np.zeros(State_dim)  
        # UAV location
        for i,uav in enumerate(self.UAVs):
            s[2*i] = uav.x                      
            s[2*i+1] = uav.y
        s[2] = self.target_loc[0]
        s[3] = self.target_loc[1]
        # s[4] = self.target_dis
        s[4] = self.data_size
        # s[5] = self.T-self.t #剩余可用时间步
        return s

    def update_state(self, engy,tar_loc,tar_dis):
        State_dim = self.uav_num*2+2+1
        s_ = np.zeros(State_dim) 
        for i,uav in enumerate(self.UAVs):
            s_[i*2] = uav.x
            s_[i*2+1] = uav.y     
        s_[2] = tar_loc[0]
        s_[3] = tar_loc[1]
        # s_[4] = tar_dis
        s_[4] = self.data_size
        # s_[5] = self.T-self.t #剩余可用时间步
        return s_

    # 奖励表达式 平滑处理
    def get_reward(self,fa,fly_engy,dis_target,dis_pre_target,dis_forward,SINR_A2G,SINR_G2A):
        """
            fa: 出界惩罚
            fly_engy: 飞行能量
            dis_target: 当前位置与目标点距离
            dis_pre_target: 当前位置与上一个目标点距离
            dis_forward: 推进距离
            SINR: 信噪比
        """
         # 前进奖励（只奖励正向前进）
        progress_reward = max(0, dis_forward * 100)
        # 停留惩罚
        stay_penalty = -10 if abs(dis_forward) < 0.05 else 0
        if self.transmit:
            stay_penalty *=2 
        
        comm_penalty = 0  
        
        
        # if SINR_A2G < self.SIR_THRESHOLD_COMM and not self.transmit:
        #     comm_penalty += self.NON_COMM_PENALTY    
        
        # if SINR_G2A < self.SIR_THRESHOLD_COVER:
        #     comm_penalty += self.NON_COVER_PENALTY
        
        total_reward = progress_reward - fa + stay_penalty + comm_penalty   
                         
        return total_reward


    def get_date_rate_A2G(self, location): #空对地
        loc_km=np.zeros(shape=(1,3))
        loc_km[0,:2]=location/10
        loc_km[0,2]=0.1
        dateRate=A2G.getPointDateRate(loc_km)
        return dateRate
    
    def get_date_rate_G2A(self, location): #地对空
        loc_km=np.zeros(shape=(1,3))
        loc_km[0,:2]=location/10
        loc_km[0,2]=0.1
        dateRate=G2A.getPointDateRate(loc_km)
        return dateRate

    def step_inside(self, actions,s,t): # 输入动作，状态值，时间步,输出下一状态和其他标志
        fa = 0.0
        reward = 0.0
        fly_energy = 0.0
        done = False
        self.t = t+1 # 时间步+1
        state_ = np.zeros(self.uav_num*2 +2+1)
        uav_location_pre = np.zeros([self.uav_num, 2])  # make a copy of the uav's location
        uav_location = np.zeros([self.uav_num, 2]) # uav current location
        for i, uav in enumerate(self.UAVs):
            uav_location_pre[i][0] = uav.x
            uav_location_pre[i][1] = uav.y
        if len(actions) == self.uav_num * 2:
            for i, uav in enumerate(self.UAVs):
                uav.move_inside(actions[0],actions[1],self.dist_max)  # execute the action执行飞行动作
            for i, uav in enumerate(self.UAVs):
                penalty, bound = self.boundary_margin(uav) # 输出出界奖励,出界bound 为 false
                fa += penalty # 累计出界奖励Pob
                if not bound: # 如果出界
                    self.fa += 1
                    uav.x = uav_location_pre[i][0]      # the uav break the boundary constraint, the action is cancelled
                    uav.y = uav_location_pre[i][1]      # 无人机突破边界约束，动作取消      
        for i, uav in enumerate(self.UAVs):
            uav_location[i][0] = uav.x
            uav_location[i][1] = uav.y

        step_cur = LA.norm(uav_location - self.target_loc)  # 此位置与目标点距离
        self.target_dis = step_cur #更新目标距离
                
        step_pre = LA.norm(uav_location_pre - self.target_loc)  # 前一位置与目标点距离
        step_cur_pre = LA.norm(uav_location - self.target_pre_loc) # 此位置与前一目标点距离         
        reduce_dis = step_pre - step_cur #推进距离
        
        MaxSINR_A2G = self.get_date_rate_A2G(uav_location)  # 空对地的最大dB
        
        MaxSINR_G2A = self.get_date_rate_G2A(uav_location)  # 地对空的最大dB
        self.data_rate = self.BandWidth * np.log2(1 + 10**(MaxSINR_A2G/10.0))
        
        #在空对地Radio Map中进行传输 
        if not self.transmit:
            # print(self.data_rate)
            self.data_size -= self.data_rate
            if self.data_size <=0:
                self.data_size = 0
                self.transmit = True
                self.trans_delay_end = self.t
                self.trans_delay += self.trans_delay_end-self.trans_delay_start
                
        # if(MaxSINR_A2G >= self.SIR_THRESHOLD_COMM ):
        #     self.data_rate = self.BandWidth * np.log2(1 + 10**(MaxSINR_A2G/10.0))  # 数据传输速率
        #     print(self.data_rate)
        #     if not self.transmit:
        #         self.data_size -= self.data_rate*self.delta_T    
        #         if(self.data_size < 20):
        #             self.transmit = True
        #             self.data_size = 0
   
        reward += self.get_reward(fa,fly_energy,step_cur,step_cur_pre,reduce_dis,MaxSINR_A2G,MaxSINR_G2A) 
                
        if step_cur<=self.distance : # 达到检查点
            #只要到目标点了，就换下一个目标点，根据传输完成情况进行奖惩
            if self.transmit:  
                print("arrive and transmit complete")
                reward += 1000      
            else:
                print("arrive but not transmit complete, data size:",self.data_size)
                reward -= self.data_size*3   
                     
                
            self.target_pre_loc = self.target_loc #更新上一个目标点的位置        
            self.transmit = False
            self.data_size = self.data_size_ini # 重置数据量  
            self.trans_delay_start = self.t
            
            if self.Traverse_order == len(self.Traverse): #到达终点了
                self.terminal = True
                done = True    
                print("=================================complete task!!========================")      
            else :                              
                self.terminal = False
                done = False               
                if self.Traverse_order < len(self.Traverse) - 1:  # 如果还有下一个目标点
                    print("=================================arrive point:",self.Traverse[self.Traverse_order])
                    self.Traverse_order += 1 #更新目标点
                    self.target = self.Users[self.Traverse[self.Traverse_order]-1]  # 更新目标点
                    self.target_loc = np.array([self.target.x, self.target.y])
                elif self.Traverse_order == len(self.Traverse) - 1:
                    print("=================================arrive point:",self.Traverse[self.Traverse_order])
                    self.Traverse_order += 1 #更新目标点
                    self.target_loc = self.end_loc
        else:
            self.terminal = False
            done = False

        if self.terminal or self.t >= self.T: # 如果任务完成，或者飞行时间步已经满了
            done = True
            if self.terminal:
                print("完成任务所用时间为：",self.t*self.delta_T)
                reward += 2000+(self.T-self.t)
            if self.t >= self.T:
                reward -= 1000
            
        state_ = self.update_state(fly_energy,self.target_loc,self.target_dis)  # 进入下一状态
        self.r = reward
        
 
        return state_, self.r,done,self.t,self.terminal

    #计算出界惩罚fa=1
    def boundary_margin(self, uav):
        """检查无人机是否出界并计算惩罚"""
        margin = 1.0  # 边界容忍度
        penalty_factor = 100 / self.uav_num
        
        # 计算到边界中心的距离
        center_x = (self.max_x + self.min_x) / 2
        center_y = (self.max_y + self.min_y) / 2
        
        # 计算超出边界的距离
        x_exceed = max(abs(uav.x - center_x) - margin * (self.max_x - self.min_x) / 2, 0.0)
        y_exceed = max(abs(uav.y - center_y) - margin * (self.max_y - self.min_y) / 2, 0.0)
        
        if x_exceed == 0.0 and y_exceed == 0.0:
            return 0.0, True  # 无惩罚，在边界内
        else:
            penalty = penalty_factor * (x_exceed**2 + y_exceed**2)
            return penalty, False  # 有惩罚，出界

    # 推进能量
    # def power(self,v):
    #     P0 = 79.8563
    #     Pi = 88.6279
    #     U_tip = 120
    #     v0 = 4.03
    #     d1 = 0.6
    #     s = 0.05
    #     rho = 1.225
    #     A = 0.503
    #     P_h = P0 * (1 + 3 * v ** 2 / U_tip ** 2) + Pi * math.sqrt(
    #         math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2)) + 0.5 * d1 * rho * s * A * v ** 3
    #     return P_h