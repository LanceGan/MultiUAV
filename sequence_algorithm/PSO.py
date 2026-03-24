import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os

class PSO(object):
    def __init__(self, num_city, data):
        self.iter_max = 500  # 迭代数目
        self.num = 200  # 粒子数目
        self.num_city = num_city  # 城市数
        self.location = data # 城市的位置坐标
        # 计算距离矩阵
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  
        # 初始化所有粒子
        self.particals = self.greedy_init(self.dis_mat,num_total=self.num,num_city =num_city)
        self.lenths = self.compute_paths(self.particals)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 记录当前的全局最优解
        self.global_best = init_path
        self.global_best_len = init_l
        # 输出解
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [init_l]
  
    def greedy_init(self, dis_mat, num_total, num_city):
        result = []
        start, end = 0, num_city - 1
        for i in range(num_total):
            rest = [x for x in range(num_city) if x != start and x != end]
            current = start
            result_one = [start]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result_one.append(end)  # 固定终点
            result.append(result_one)
        return result

    def random_init(self, num_total, num_city):
        result = []
        start, end = 0, num_city - 1
        middle = [x for x in range(num_city) if x != start and x != end]
        for _ in range(num_total):
            random.shuffle(middle)
            result.append([start] + middle + [end])
        return result

    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat
  
    def compute_pathlen(self, path, dis_mat):
        result = 0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    def cross(self, cur, best):
        one = cur.copy()
        start, end = one[0], one[-1]
        middle = one[1:-1]

        l = list(range(len(middle)))
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        cross_part = best[1:-1][x:y]

        tmp = []
        for t in middle:
            if t in cross_part:
                continue
            tmp.append(t)

        one1 = [start] + tmp + cross_part + [end]
        one2 = [start] + cross_part + tmp + [end]
        l1 = self.compute_pathlen(one1, self.dis_mat)
        l2 = self.compute_pathlen(one2, self.dis_mat)

        if l1 < l2:
            return one1, l1
        else:
            return one2, l2

    def mutate(self, one):
        one = one.copy()
        start, end = one[0], one[-1]
        middle = one[1:-1]
        l = list(range(len(middle)))
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        middle[x], middle[y] = middle[y], middle[x]
        mutated = [start] + middle + [end]
        l2 = self.compute_pathlen(mutated, self.dis_mat)
        return mutated, l2

    def pso(self):
        for cnt in range(1, self.iter_max):
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                new_one, new_l = self.cross(one, self.global_best)
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                    
                one, tmp_l = self.mutate(one)
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                self.particals[i] = one
                self.lenths[i] = tmp_l
                
            self.eval_particals()
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        return self.location[best_path], best_length

# ================= 工具函数 =================

def read_clustered_data(coords_path, cluster_path):
    """同时读取坐标和聚类结果，返回按簇划分的数据字典"""
    coords = []
    with open(coords_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                coords.append([float(x) for x in parts])
    coords = np.array(coords)
    
    clusters = []
    with open(cluster_path, 'r') as f:
        for line in f:
            val = line.strip()
            if val:
                clusters.append(int(val))
    clusters = np.array(clusters)
    
    unique_clusters = np.unique(clusters)
    clustered_data = {}
    for cid in unique_clusters:
        clustered_data[cid] = coords[clusters == cid]
        
    return clustered_data


if __name__ == '__main__':
    User_num = 20
    Cluster_num = 2
    
    # 文件路径
    coords_file = '../results/datas/Users_%d.txt' % User_num
    # cluster_file = 'results/datas/cluster/Users_%d_ClusteredUAV_%d.txt' % (User_num, Cluster_num)
    cluster_file = '../results/datas/cluster/Users_%d_Clustered_comm_4DUAV_%d.txt' % (User_num, Cluster_num)
    save_path = '../results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_PSO_%d.npz' % (User_num, Cluster_num)
    
    # 起点和终点坐标设定 (扩展为3维以匹配数据)
    
    if Cluster_num == 2:
        #uav = 2
        ini_loc = np.array([14.76, 14.83, 0.0])
        end_loc = np.array([27.62, 23.47, 0.0])
    elif Cluster_num == 3:
        #uav = 3
        ini_loc = np.array([32.88, 22.67, 0.0])
        end_loc = np.array([21.62, 48.47, 0.0])
    elif Cluster_num == 4:
        #uav = 4
        ini_loc = np.array([34.12, 28.79, 0.0])
        end_loc = np.array([38.46, 45.23, 0.0])

    # 🔥 修改点1：额外读取真实的聚类标签，提取全局索引
    cluster_labels = []
    with open(cluster_file, 'r') as f:
        for line in f:
            if line.strip():
                cluster_labels.append(int(line.strip()))
    cluster_labels = np.array(cluster_labels)
    
    # 读取并按簇分组数据
    clustered_data = read_clustered_data(coords_file, cluster_file)
    
    res_indices = {}
    res_paths = {} # 用于存储最终带有真实坐标的完整路径，方便画图
    
    # 初始化画布，全白背景
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    colors = ['blue', 'green', 'purple', 'brown'] 
    
    # 如果有A2G的Radio Map，可以叠加在底部
    # try:
    #     npzfile_sinr = np.load('results/datas/radiomap/Radio_datas_A2G.npz')
    #     OutageMapActual = npzfile_sinr['arr_0'] 
    #     Y_vec2 = npzfile_sinr['arr_2']  
    #     X_vec2 = npzfile_sinr['arr_3']
    #     plt.contourf(np.array(Y_vec2) * 10, np.array(X_vec2) * 10, 1-OutageMapActual)
    #     v = np.linspace(-10, 30, 6, endpoint=True)
    #     cbar = plt.colorbar(ticks=v)
    #     cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
    # except FileNotFoundError:
    #     print("未找到Radio Map文件，跳过底层图渲染。")

    print("\n================ 开始 PSO 路径规划 ================")
    for cid, data in clustered_data.items():
        print(f"--- 正在为 簇(UAV) {cid} 规划路径 ---")
        
        # 将起点、任务点和终点拼接
        full_data = np.vstack([ini_loc, data, end_loc])
        num_nodes = full_data.shape[0]
        
        # 运行 PSO
        pso = PSO(num_city=num_nodes, data=full_data.copy())
        Best_path_coords, Best_length = pso.run()
        
        # 保存真实坐标路径用于绘制
        res_paths[cid] = Best_path_coords
        print(f"簇 {cid} 规划完成，最佳路径长度: {Best_length:.2f}")
        
        # 提取索引以供原有的逻辑保存 (剥离起点和终点)
        best_indices_in_fulldata = pso.best_path
        
        # 🔥 修改点2：获取当前簇的全局真实索引
        global_indices = np.where(cluster_labels == int(cid))[0]
        
        orig_order = []
        for idx in best_indices_in_fulldata:
            if idx != 0 and idx != num_nodes - 1:
                local_idx = idx - 1  # 减1是因为排除了最前面的起点
                global_idx = int(global_indices[local_idx]) # 映射回全局索引
                orig_order.append(global_idx)
                
        res_indices[int(cid)] = orig_order
        
        # 绘制当前无人机的路径
        color = colors[int(cid) % len(colors)]
        
        # 画任务点 (避开起终点)
        plt.scatter(Best_path_coords[1:-1, 0] * 1000, Best_path_coords[1:-1, 1] * 1000, 
                    c=color, marker='^', s=80, edgecolors='white', label=f'UAV {cid} Targets')
        # 画连线
        plt.plot(Best_path_coords[:, 0] * 1000, Best_path_coords[:, 1] * 1000, 
                 '-', c=color, linewidth=2, label=f'UAV {cid} Path')

    # 保存原始的聚类顺序结果
    np.savez(save_path, result=res_indices)
    print(f"\n路径序列结果已保存至: {save_path}")

    # 绘制公共的起飞点与降落点
    plt.scatter(ini_loc[0] * 1000, ini_loc[1] * 1000, color='orange', s=200, marker='o', edgecolors='black', label='Start', zorder=5)
    plt.scatter(end_loc[0] * 1000, end_loc[1] * 1000, color='red', s=200, marker='*', edgecolors='black', label='End', zorder=5)
    
    # 图像细节设置
    plt.xlabel('x (meter)', fontsize=14)    
    plt.ylabel('y (meter)', fontsize=14)
    plt.title('PSO Paths for Clustered UAVs', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 确保图片保存不带任何透明通道，强制白底
    if not os.path.exists('../results/figs/sequence/'):
        os.makedirs('../results/figs/sequence/')
    fig_save_path = '../results/figs/sequence/PSO_Clustered_Paths_UAV_%d.png' % Cluster_num
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    print(f"路径可视化图片已保存至: {fig_save_path}")
    plt.show()