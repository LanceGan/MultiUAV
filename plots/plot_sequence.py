from matplotlib import pyplot as plt
import numpy as np 
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
    
if __name__ == '__main__':

    User_num = 20
    Cluster_num = 2
    ini_loc = [[14.76, 14.83],[32.88, 22.67],[34.12, 28.79]][0]
    end_loc = [[27.62, 23.47],[21.62, 48.47],[38.46, 45.23]][0]
    
    Cluster_result = 'results/datas/cluster/Users_%d_Clustered'% User_num+'UAV_'+str(Cluster_num)+'.txt'
    Cluster_label = load_cluster(Cluster_result,User_num)

    Users_path = 'results/datas/Users_%d.txt' % User_num
    x0_user, y0_user = load_user(Users_path,User_num)
    
    save_path = 'results/datas/sequence/Users_%d_Clusteredsave_path_Path'% User_num+'UAV_'+str(Cluster_num)
    points = np.column_stack((x0_user, y0_user))
    
    # 将坐标转换为千米
    points = points*100
    ini_loc = np.array(ini_loc) *100
    end_loc = np.array(end_loc) *100

    npz_data = np.load(save_path + '.npz', allow_pickle=True)
    result = npz_data['result'].item()  # 读取字典
    print("最终路径结果（按簇划分）:", result)
    
    # 可视化路径
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', Cluster_num)  # 获取颜色映射
    
    # 绘制每个簇的路径，包含起点与终点（起/终点由 ini_loc/end_loc 提供）
    for c in range(Cluster_num):
        idxs = result[c]
        if len(idxs) == 0:
            continue
        cluster_idxs = np.array(idxs, dtype=int)
        cluster_points = points[cluster_idxs]
        # 将起点、簇内路径点、终点连成一条路径用于绘制
        path_points = np.vstack([ini_loc, cluster_points])
        plt.plot(path_points[:, 0], path_points[:, 1], marker='o', color=colors(c), label=f'Cluster {c}')
        # 仅为簇内点添加索引标签，起点/终点用统一符号标注（后面已有绘制）
        for i, point in enumerate(cluster_points):
            plt.text(point[0], point[1], str(cluster_idxs[i]), fontsize=9, ha='right')
    
    plt.plot(ini_loc[0], ini_loc[1], 'r^', markersize=12, label='Start')
    plt.plot(end_loc[0], end_loc[1], 'gs', markersize=12, label='End')
    plt.legend()
    plt.grid()
    save_path_fig = 'results/figs/sequence/Users_%d_Clusteredsave_path_Path'% User_num+'UAV_'+str(Cluster_num)+'.png'
    plt.savefig(save_path_fig, dpi=300, bbox_inches='tight', format='png')
    plt.show()  
    