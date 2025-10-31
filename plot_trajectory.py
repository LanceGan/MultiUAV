from matplotlib import pyplot as plt
import numpy as np 
uav_num = 2
user_num = 20 
users_path = r'results/datas/Users_20.txt'
trajectory_file = r'results/trajectory/MultiUAV_uav2_ep9.npz'
f = open(users_path, 'r')
x0_user = []
y0_user = []
if f:
    for j in range(user_num):
        user_loc = f.readline()
        # print("user_loc", user_loc)
        user_loc = user_loc.split(' ')
        x0_user.append(float(user_loc[0]))
        # print("x_user",x_user)
        y0_user.append(float(user_loc[1]))
        
#加载轨迹数据
    print(f"\n加载轨迹文件: {trajectory_file}")
    try:
        traj_data = np.load(trajectory_file, allow_pickle=True)
        print("✓ 轨迹加载成功")
    except Exception as e:
        print(f"✗ 轨迹加载失败: {e}")
    
    x_uav = traj_data['x_uav']  # [uav_num, steps]
    y_uav = traj_data['y_uav']
    x_user = traj_data['x_user']
    y_user = traj_data['y_user']
    uav_num = x_uav.shape[0]
    steps = traj_data['steps']
    
    print(f"  无人机数量: {uav_num}")
    print(f"  总步数: {steps}")
    print(f"  完成目标: {traj_data['completed_targets']}")

        
if __name__ == '__main__':
    
    Radio_Map = 'G2A'
    #画Radio Map 
    fig_1 = plt.figure(30)
    if Radio_Map == 'A2G':
        npzfile = np.load('results/datas/Radio_datas_A2G.npz')
        OutageMapActual = npzfile['arr_0']
        OutageMapActual_SINR = npzfile['arr_1']
        X_vec = npzfile['arr_2']  # [0,1....100]标号
        Y_vec = npzfile['arr_3']
        plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, OutageMapActual_SINR)
        v = np.arange(-20, 36, 4)
        cbar = plt.colorbar(ticks=v)
        cbar.set_label('SNR', labelpad=20, rotation=270, fontsize=14)
    elif Radio_Map =='G2A':
        npzfile = np.load('results/datas/Radio_datas.npz')
        OutageMapActual = npzfile['arr_0']
        OutageMapActual_SINR = npzfile['arr_1']
        X_vec = npzfile['arr_2']  # [0,1....100]标号
        Y_vec = npzfile['arr_3']
        plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual)
        v = np.linspace(0, 1.0, 11, endpoint=True)
        cbar = plt.colorbar(ticks=v)
        cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)

    
    for i in range(user_num): # 画用户位置
        if i!=0 and i!= user_num-1:
            if i == 1 :
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c='black', marker='^',s=60,label='Inspection Point')
            else :
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c='black', marker='^',s=60)
        else :
            if i == 0:
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c='red', marker='o',s=60,label='Start') #画起始点
            else :
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c='orange', marker='o',s=60,label='End') #画终点
    
    #绘制无人机轨迹
    colors = ['cyan', 'lime', 'yellow', 'magenta', 'white', 
              'lightblue', 'lightgreen', 'pink']
    
    for i in range(uav_num):
        color = colors[i % len(colors)]
        # 轨迹线 (将坐标转换为米)
        x_traj = x_uav[i][:steps+1] * 100
        y_traj = y_uav[i][:steps+1] * 100
        plt.plot(x_traj, y_traj, color=color, label=f'UAV {i}')
        
    plt.legend(fontsize = 6)
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    
    
    # plt.title(title_str + f" ({Radio_Map})", fontsize=16)
    # save_path = 'results/figs/'+str(method)+'_trajectory_'+str(Radio_Map)+'.pdf' 
    # plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path = 'results/figs/'+str(user_num)+'_trajectory_'+str(Radio_Map)+'.pdf' 
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    