from matplotlib import pyplot as plt
import numpy as np 
user_num = 20 
users_path = 'results/datas/Users_20.txt'
trajectory_file = r'results/trajectory/MultiUAV_uav2_ep9.npz'
ini_loc = [14.76, 14.83]
end_loc = [27.62, 23.47]

def load_user(path,user_num):
    f = open(path, 'r')
    x = []
    y = []
    for j in range(user_num):
        user_loc = f.readline()
        # print("user_loc", user_loc)
        user_loc = user_loc.split(' ')
        x.append(float(user_loc[0]))
        # print("x_user",x_user)
        y.append(float(user_loc[1]))
    f.close()
    return x, y
    print(f"✓ 用户位置加载成功，共加载{user_num}个用户位置")

def load_cluster(path,user_num):
    f = open(path, 'r')
    cluster_result = []
    for j in range(user_num):
        cluster_label = f.readline()
        cluster_result.append(int(cluster_label))
    f.close()
    return cluster_result
    print(f"✓ 用户聚类结果加载成功，共加载{user_num}个用户聚类标签")
        
def load_trajectory(trajectory_file):
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
    return x_uav, y_uav, x_user, y_user, uav_num, steps

        
if __name__ == '__main__':
    
    x0_user, y0_user = load_user(users_path,user_num)
    mode = ('user','cluster','trajectory')[1]  
    mode_add = ('origin','comm')[1]
    UAV_NUM = 4
    
    # 加载聚类结果
    if mode_add == 'origin':
        cluster_result = 'results/datas/Users_20_Clustered'+'UAV_'+str(UAV_NUM)+'.txt'
    elif mode_add == 'comm':
        cluster_result = 'results/datas/Users_20_Clustered_comm'+'UAV_'+str(UAV_NUM)+'.txt'
    cluster_label = load_cluster(cluster_result,user_num)
    
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

    # 画用户位置
    if mode == 'user':
        for i in range(user_num): 
            if i == 0 :
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c='black', marker='^',s=60,label='Inspection Point')
            else :
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c='black', marker='^',s=60)
    
    # 画聚类结果
    if mode == 'cluster':
        colors = ['blue', 'magenta', 'orange', 'purple', 'cyan', 'brown', 'pink']
        seen = set()
        for i in range(user_num):
            cluser_idx = cluster_label[i]
            color = colors[int(cluser_idx) % len(colors)]
            if cluser_idx not in seen:
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c=color, marker='^',s=30, label=f'Cluster {cluser_idx}')
                seen.add(cluser_idx)
            else:
                plt.scatter(x0_user[i]*100, y0_user[i]*100, c=color, marker='^',s=30)
    
    # 画起点和终点
    plt.scatter(ini_loc[0]*100, ini_loc[1]*100, c='red', marker='o',s=60,label='Start')
    plt.scatter(end_loc[0]*100, end_loc[1]*100, c='yellow', marker='s',s=60,label='End')
                
    
    #绘制无人机轨迹
    if mode == 'trajectory':
        colors = ['blue', 'magenta', 'white', 
                'lightblue', 'lightgreen', '']
        x_uav, y_uav, x_user, y_user, uav_num, steps = load_trajectory(trajectory_file)
        
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
    if mode == 'cluster':
        save_path = 'results/figs/'+'UAV_'+str(UAV_NUM)+'_'+str(mode_add)+'_'+str(Radio_Map)+'.png' 
    
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    