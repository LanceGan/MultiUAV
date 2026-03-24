import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class GA(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.05
        
        self.dis_mat = self.compute_dis_mat(num_city, data)
        # 初始化种群
        self.fruits = self.greedy_init(self.dis_mat, num_total, num_city)
        
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]
        
        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]

    def random_init(self, num_total, num_city):
        """随机初始化，锁定起点0和终点num_city-1"""
        result = []
        middle = [x for x in range(1, num_city - 1)]
        for i in range(num_total):
            random.shuffle(middle)
            result.append([0] + middle.copy() + [num_city - 1])
        return result

    def greedy_init(self, dis_mat, num_total, num_city):
        """贪心初始化，锁定起点0和终点num_city-1"""
        result = []
        for i in range(num_total):
            rest = [x for x in range(1, num_city - 1)]
            current = 0
            result_one = [0]
            
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    # 加入随机扰动避免所有初始个体完全一样
                    if dis_mat[current][x] < tmp_min and np.random.rand() > 0.1:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                
                if tmp_choose != -1:
                    current = tmp_choose
                    rest.remove(tmp_choose)
                    result_one.append(tmp_choose)
                else:
                    # 退避策略
                    choose = rest.pop()
                    result_one.append(choose)
                    current = choose
            
            result_one.append(num_city - 1)
            result.append(result_one)
            
        return result

    def compute_dis_mat(self, num_city, location):
        """计算距离矩阵"""
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
        """计算路径长度"""
        result = 0.0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_adp(self, fruits):
        """计算适应度（路径长度的倒数）"""
        adp = []
        for fruit in fruits:
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def ga_cross(self, x, y):
        """交叉操作（PMX），锁定首尾"""
        len_ = len(x)
        assert len(x) == len(y)
        
        # 只在中间的任务点进行交叉切片，保护首尾
        path_list = list(range(1, len_ - 1))
        if len(path_list) < 2:
            return list(x), list(y)
            
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (start <= index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (start <= index < end):
                y_confict_index.append(index)

        # 交叉片段
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        """选择父代"""
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        """轮盘赌选择"""
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        index1, index2 = 0, 0
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        """变异操作（逆序截取部分片段），锁定首尾"""
        len_ = len(gene)
        path_list = list(range(1, len_ - 1))
        if len(path_list) < 2:
            return list(gene)
            
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        scores = self.compute_adp(self.fruits)
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        
        fruits = parents.copy()
        
        # 增加防死循环计数器
        stuck_counter = 0 
        
        while len(fruits) < self.num_total:
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
                
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            
            # 正常逻辑：尽量要求新生成的个体不与种群已有基因重复
            if stuck_counter < 50:
                if x_adp > y_adp and (gene_x_new not in fruits):
                    fruits.append(gene_x_new)
                    stuck_counter = 0
                elif x_adp <= y_adp and (gene_y_new not in fruits):
                    fruits.append(gene_y_new)
                    stuck_counter = 0
                else:
                    stuck_counter += 1 # 产生重复个体，累加计数器
            else:
                # 触发防死循环机制：若某簇任务点极少，组合数耗尽，则允许种群存入重复个体以完成繁衍
                if x_adp > y_adp:
                    fruits.append(gene_x_new)
                else:
                    fruits.append(gene_y_new)
                stuck_counter = 0 # 存入后重置计数器

        self.fruits = fruits
        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
                
        return self.location[BEST_LIST], 1. / best_score, BEST_LIST


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


if __name__ == "__main__":
    User_num = 20
    Cluster_num = 2
    
    # 文件路径
    coords_file = '../results/datas/Users_%d.txt' % User_num
    # cluster_file = 'results/datas/cluster/Users_%d_ClusteredUAV_%d.txt' % (User_num, Cluster_num)
    cluster_file = '../results/datas/cluster/Users_%d_Clustered_comm_4DUAV_%d.txt' % (User_num, Cluster_num)
    save_path = '../results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_GA_%d.npz' % (User_num, Cluster_num)
    
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

    # 1. 额外读取真实的聚类标签，提取全局索引
    cluster_labels = []
    with open(cluster_file, 'r') as f:
        for line in f:
            if line.strip():
                cluster_labels.append(int(line.strip()))
    cluster_labels = np.array(cluster_labels)

    # 读取并按簇分组数据
    clustered_data = read_clustered_data(coords_file, cluster_file)
    res_indices = {}
    
    # 初始化画布，全白背景
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    colors = ['blue', 'green', 'purple', 'brown'] 
    
    # 叠加 Radio Map 
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

    print("\n================ 开始 GA 路径规划 ================")
    for cid, data in clustered_data.items():
        print(f"--- 正在为 簇(UAV) {cid} 规划路径 ---")
        
        # 将起点、任务点和终点拼接
        full_data = np.vstack([ini_loc, data, end_loc])
        num_nodes = full_data.shape[0]
        
        # 运行 GA
        model = GA(num_city=num_nodes, num_total=100, iteration=300, data=full_data.copy())
        Best_path_coords, Best_length, Best_indices = model.run()
        
        print(f"簇 {cid} 规划完成，最佳路径长度: {Best_length:.2f}")
        
       # 2. 获取当前簇的全局真实索引并进行映射
        global_indices = np.where(cluster_labels == int(cid))[0]
        
        orig_order = []
        for idx in Best_indices:
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
    if not os.path.exists('results/datas/sequence/'):
        os.makedirs('results/datas/sequence/')
    np.savez(save_path, result=res_indices)
    print(f"\n路径序列结果已保存至: {save_path}")

    # 绘制公共的起飞点与降落点
    plt.scatter(ini_loc[0] * 1000, ini_loc[1] * 1000, color='orange', s=200, marker='o', edgecolors='black', label='Start', zorder=5)
    plt.scatter(end_loc[0] * 1000, end_loc[1] * 1000, color='red', s=200, marker='*', edgecolors='black', label='End', zorder=5)
    
    # 图像细节设置
    plt.xlabel('x (meter)', fontsize=14)    
    plt.ylabel('y (meter)', fontsize=14)
    plt.title('GA Paths for Clustered UAVs', fontsize=16)
    plt.legend()
    
    # 取消背景网格线，并确保图像四个边界线都要有
    ax = plt.gca()
    ax.grid(False) 
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # 保存绘图：确保全白背景
    if not os.path.exists('../results/figs/sequence/'):
        os.makedirs('../results/figs/sequence/')
    fig_save_path = '../results/figs/sequence/GA_Clustered_Paths_UAV_%d.png' % Cluster_num
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    print(f"路径可视化图片已保存至: {fig_save_path}")
    plt.show()