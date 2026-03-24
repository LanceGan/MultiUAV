# 绘制聚类结果图，仅包含散点，无连线
import os
from matplotlib import pyplot as plt
import numpy as np 

def load_user(path, user_num):
    f = open(path, 'r')
    x = []
    y = []
    for j in range(user_num):
        user_loc = f.readline()
        user_loc = user_loc.split(' ')
        x.append(float(user_loc[0]))
        y.append(float(user_loc[1]))
    f.close()
    print(f"✓ 用户位置加载成功，共加载{user_num}个用户位置")
    return x, y

def load_cluster(path, user_num):
    f = open(path, 'r')
    cluster_result = []
    for j in range(user_num):
        cluster_label = f.readline()
        cluster_result.append(int(cluster_label))
    f.close()
    print(f"✓ 用户聚类结果加载成功，共加载{user_num}个用户聚类标签")
    return cluster_result
    
if __name__ == '__main__':

    User_num = 40
    Cluster_num = 4
    
    # 提取起点和终点坐标
    # uav=2
    # ini_loc = [14.76, 14.83]
    # end_loc = [27.62, 23.47]
    # uav=3
    # ini_loc = [32.88, 22.67]
    # end_loc = [21.62, 48.47]
    # uav=4
    ini_loc = [34.12, 28.79]
    end_loc = [38.46, 45.23]
    
    
    # 加载聚类标签
    # Cluster_result = 'results/datas/cluster/Users_%d_ClusteredUAV_%d.txt' % (User_num, Cluster_num)
    Cluster_result = 'results/datas/cluster/Users_40_Clustered_comm_4DUAV_4.txt'
    Cluster_label = load_cluster(Cluster_result, User_num)
    Cluster_label = np.array(Cluster_label)

    # 加载用户位置
    Users_path = 'results/datas/Users_%d.txt' % User_num
    x0_user, y0_user = load_user(Users_path, User_num)
    
    points = np.column_stack((x0_user, y0_user))
    
    # 将坐标转换为千米比例
    points = points * 100
    ini_loc = np.array(ini_loc) * 100
    end_loc = np.array(end_loc) * 100

    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', Cluster_num)  # 获取颜色映射
    
    # 绘制每个簇的散点（不画连线）
    for c in range(Cluster_num):
        # 筛选属于当前簇的点的索引
        cluster_idxs = np.where(Cluster_label == c)[0]
        if len(cluster_idxs) == 0:
            continue
            
        cluster_points = points[cluster_idxs]
        
        # 使用 scatter 绘制散点
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(c), marker='o', s=60, label=f'Cluster {c}')
        
        # 为簇内点添加索引标签
        for i, point in enumerate(cluster_points):
            plt.text(point[0], point[1], str(cluster_idxs[i]), fontsize=10, ha='right', va='bottom')
    
    # 绘制起点和终点
    plt.plot(ini_loc[0], ini_loc[1], 'r^', markersize=14, label='Start')
    plt.plot(end_loc[0], end_loc[1], 'gs', markersize=14, label='End')
    
    plt.legend()
    plt.grid(True)
    
    # 确保保存目录存在
    save_dir = 'results/figs/cluster'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 保存结果图（文件名用 User_num 和 Cluster_num/UAV_num 区分）
    save_path_fig = f'{save_dir}/Users_{User_num}_UAV_{Cluster_num}.png'
    plt.savefig(save_path_fig, dpi=300, bbox_inches='tight', format='png')
    print(f"✓ 聚类结果图已保存至: {save_path_fig}")
    
    plt.show()