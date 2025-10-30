import torch
import torch.nn as nn
import torch.nn.functional as F
from World import World
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import os
from torch.utils.tensorboard import SummaryWriter


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def draw_location_3d(x_uav, y_uav,z_uav,t, x_user, y_user,z_user, savepath,s):
    print("time:",t)
    x_uav = np.transpose(x_uav) # 6.33 m
    y_uav = np.transpose(y_uav)
    h_uav = np.transpose(z_uav)
    x_uav =  x_uav*100 # 转化为m
    y_uav =  y_uav*100
    h_uav =  h_uav*100

    fig_1 = plt.figure(30)
    npzfile = np.load('results/datas/Radio_datas.npz')
    OutageMapActual = npzfile['arr_0']
    X_vec = npzfile['arr_2']  # [0,1....100]标号
    Y_vec = npzfile['arr_3']

    plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual)
    v = np.linspace(0, 1.0, 11, endpoint=True)
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)


    for i in range(user_num): # 画用户位置
        plt.scatter(x_user[i]*100, y_user[i]*100, c='black', marker='^')
    for i in range(uav_num): # 画无人机轨迹
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], c='blue')
        plt.plot(x_uav[i][t], y_uav[i][t], c='green', marker='.',markersize=8)
    # new_ticks = np.linspace(0, 10, 6)
    # plt.xticks(new_ticks,['0','200','400','600','800','1000'])
    # plt.yticks(new_ticks,['0','200','400','600','800','1000'])
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    # plt.axis('equal')
    plt.savefig(savepath)
    # plt.show()
    plt.close()

  

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, maxaction):
		super(Actor, self).__init__()
		#self.fc_t_sprend = nn.Linear(1, user_num) # 信息素维度扩展至 user num
		#self.fc_loc_sprend = nn.Linear(2, user_num) # 位置维度扩展
		self.l1 = nn.Linear(state_dim, net_width) # 4k
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, net_width)
		self.l4 = nn.Linear(net_width, action_dim)

		self.maxaction = maxaction

	def forward(self, state):
		#t = self.fc_t_sprend(state[:, -1:])
		#loc = self.fc_loc_sprend(state[:, :]) # 位置
		#a = torch.cat([t, loc, state[:, :-4]], 1) # 除了位置和信息素 len(state[:, :-4])=19
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		a = torch.tanh(self.l3(a))
		a = torch.tanh(self.l4(a)) * self.maxaction
		return a

############### test
uav_num = 1
user_num = 6
uav_h = 1
result_path = 'Test_/'
mkdir(result_path[:-1])
mkdir('results/trajectory')
T = 1000
Length = 20   #1km
Width = 20
s_dim = uav_num*2 +2 +1 #[uav.x,uav.y,tar.x,tar.y,data_size]
a_dim = uav_num*2
V_max = 0.50 # 最大速度
delta_t = 0.5 # 时隙长度
dist_max =V_max * delta_t
max_action = np.array([math.pi,dist_max])
test_episode = 6

data_size = 200

"""
 [ 8.22  4.34]
 [ 6.5  11.43]
 [ 4.3  17.9 ]
 [15.9  17.1 ]
 [11.34 12.8 ]
 [14.52  7.05]
 [18.7  11.7 ]]
"""
# ini_loc = [3.53,8.45] # version2
# end_loc = [19.23,4.22]
ini_loc = [2.62,2.65] # version1
end_loc = [18.95,11.12]
BS_loc = np.array([[4.77,4.56,0.25],[14.89,4.89,0.25],[4.63,13.78,0.25],[13.48,14.89,0.25]])
# trave_order = [1,6,5,2,3,4]#Ours
# trave_order = [2,3,4,5,1,6]#GA
trave_order = [1,2,3,4,5,6]#PSO
# trave_order = [6,5,4,3,2,1]#ACO
world = World(length=Length, width=Width, uav_num=uav_num, user_num=user_num, delta_t=delta_t,dist_max=dist_max,t=T, uav_h=uav_h,
              data_size=data_size, ini_loc=ini_loc, end_loc=end_loc,
              users_name='results/datas/Users_'+str(user_num)+'.txt'
              , BS_loc=BS_loc,traverse_sequence=trave_order)
model = Actor(s_dim, a_dim, 256, max_action)

model.load_state_dict(torch.load('results/models/300MB_PSO.pth')) # map_location={'cuda:1':'cuda:0'})
model.eval()

success_rate = 0.0
Complete_time = np.zeros(test_episode)
fly_time = 0
Energy = np.zeros(test_episode)
Distance = np.zeros(test_episode)
Fly_v = np.zeros(test_episode)
Out_time = np.zeros(test_episode)
FLY_V_ep = np.zeros([test_episode, T + 1, world.uav_num])
Data_size = np.zeros([test_episode, T + 1, world.uav_num])
Data_rate = np.zeros([test_episode, T + 1, world.uav_num])
x0_user = np.zeros( world.user_num)
y0_user = np.zeros( world.user_num)
z0_user = np.zeros( world.user_num)
x0_uav = np.zeros([test_episode, T + 1, world.uav_num])
y0_uav = np.zeros([test_episode, T + 1, world.uav_num])
z0_uav = np.zeros([test_episode, T + 1, world.uav_num])
location = np.zeros([3, world.uav_num])

def power(v):
    P0 = 79.8563
    Pi = 88.6279
    U_tip = 120
    v0 = 4.03
    d1 = 0.6
    s = 0.05
    rho = 1.225
    A = 0.503
    P_h = P0 * (1+3*v**2/U_tip**2)+Pi*math.sqrt(math.sqrt(1+v**4/(4*v0**4))-v**2/(2*v0**2))+0.5*d1*rho*s*A*v**3
    return P_h

total_time = 0.0
total_delay = 0.0 
data_total = 0.0
for episode in range(test_episode):
    # s, t = world.reset()
    s, t = world.reset()
    fly_dis = 0.0
    move_time = 0.0
    fly_energy = 0.0
    fly_v_avg = 0.0
    fly_v = 0.0

    for i, uav in enumerate(world.UAVs):
        location[0][i] = uav.x
        location[1][i] = uav.y
        location[2][i] = uav.h
    x0_uav[episode][0] = location[0]
    y0_uav[episode][0] = location[1]
    z0_uav[episode][0] = location[2]
    num_fa2 = 0
    sum_reward = 0
    actions = np.zeros(2 * uav_num)
    done = False
    while not done:
        s_shape = torch.unsqueeze(torch.FloatTensor(s), 0)
        with torch.no_grad():
            actions = model(s_shape)[0].detach().numpy()
            actions[0] = np.clip(actions[0], -np.pi, np.pi)      # 角度
            actions[1] = np.clip(actions[1], 0, dist_max)        # 距离
        fly_v_avg += 100 * actions[1]/delta_t # avg_v
        fly_v = 100 * actions[1]/delta_t # every step v
        fly_dis = fly_dis + actions[1] 
        fly_energy += power(actions[1]*100/delta_t)*delta_t
        # print("速度：",fly_v)
        s_, r, done, t,terminal = world.step_inside(actions,s, t)
        sum_reward += r
        data_total += world.data_rate
        move_time += 1.0
        s = s_
        for i, uav in enumerate(world.UAVs):
            location[0][i] = uav.x
            location[1][i] = uav.y
            location[2][i] = uav.h
        for i, user in enumerate(world.Users):
            x0_user[i] = user.x
            y0_user[i] = user.y
            # print(uav.x,uav.y,uav.h)
        x0_uav[episode][t] = location[0]
        y0_uav[episode][t] = location[1]
        z0_uav[episode][t] = location[2]
        FLY_V_ep[episode][t] = fly_v
        Data_size[episode][t] = world.data_size
        Data_rate[episode][t] = world.data_rate
        #writer.add_scalar('sum_reward%s'% str(episode),sum_reward, t)
        if done:  # and episode % 5 == 0:
            total_time += world.t
            total_delay += world.trans_delay
            # print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'reward: ', r, 'fa', world.fa,
            #       'raw_r', world.r)
            # print('Energy:',world.total_engy,"avg_fly_v",fly_v_avg/t,'out time:',world.out_time,'world datasize:',world.data_size,"total_data:",data_total)
    
    np.savez('results/trajectory/PSO_200MB.npz' , x0_uav=x0_uav[episode], y0_uav=y0_uav[episode], T = t)
    
    draw_location_3d(x0_uav[episode], y0_uav[episode],z0_uav[episode], t, x0_user, y0_user,z0_user,
                result_path + 'Test_UAVPath_Users_%s.jpg' % str(episode).zfill(2), s)
    if world.terminal:
        success_rate += 1.0
    Complete_time[episode] = t
    fly_time += t
    Energy[episode] = fly_energy
    Distance[episode] = fly_dis
    Fly_v[episode] = fly_v_avg/t
    Out_time[episode]=world.out_time
    avg_out_time = sum(Out_time)/test_episode
    print(move_time)

print("任务完成时间:",total_time/test_episode)
print("传输时延:",total_delay/test_episode)
print('中断时间:',avg_out_time)
print("总吞吐量:",data_total/test_episode)

# avg_time = fly_time/test_episode
# avg_engy = sum(Energy)/test_episode
# avg_length = sum(Distance)/test_episode
# avg_v = sum(Fly_v)/test_episode
# print('###Energy',Energy)
# print('##', sum(Complete_time)/test_episode)
# print('###engy',sum(Energy)/test_episode)
# print('Distance:', sum(Distance)/test_episode)
# print('fly_time: ',fly_time/test_episode)

# np.savez(r"D:\GraduPaper_zw\newBS\final\data_process\data\Proposed_TD3_[3]_[TSP]"+str(data_size),avg_time,avg_length,avg_engy,avg_v,FLY_V_ep,avg_out_time, Data_size,Data_rate,x0_uav, y0_uav, z0_uav)
# np.savez("D:\GraduPaper_zw\分段优化轨迹\only_ending_new\绘图\data\path_nodisend_250mb", x0_uav, y0_uav, z0_uav) # 保存轨迹
