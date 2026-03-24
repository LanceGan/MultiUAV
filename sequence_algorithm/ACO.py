import math
import numpy as np
import matplotlib.pyplot as plt
import os


class ACO(object):
    def __init__(self, num_city, data, start_node=0, end_node=None):
        self.m = 50  # 蚂蚁数量
        self.alpha = 1  # 信息素重要程度因子
        self.beta = 5  # 启发函数重要因子
        self.rho = 0.1  # 信息素挥发因子
        self.Q = 1  # 常量系数
        self.num_city = num_city  # 城市规模
        self.location = data  # 城市坐标
        self.Tau = np.zeros([num_city, num_city])  # 信息素矩阵
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]  # 生成的蚁群
        self.iter = 1
        self.iter_max = 500
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵
        self.Eta = 10. / self.dis_mat  # 启发式函数
        self.paths = None  # 蚁群中每个个体的长度
        # 存储存储每个温度下的最终路径，画出收敛图
        self.iter_x = []
        self.iter_y = []
        
        self.start_node = start_node
        self.end_node = end_node if end_node is not None else start_node
        # self.greedy_init(self.dis_mat,100,num_city)
    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
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
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        result = result[index]
        for i in range(len(result)-1):
            s = result[i]
            s2 = result[i+1]
            self.Tau[s][s2]=1
        self.Tau[result[-1]][result[0]] = 1
        # for i in range(num_city):
        #     for j in range(num_city):
        # return result

    # 轮盘赌选择
    def rand_choose(self, p):
        x = np.random.rand()
        for i, t in enumerate(p):
            x -= t
            if x <= 0:
                break
        return i

    def get_ants(self, num_city):
        for i in range(self.m):
            self.Table[i][0] = self.start_node
            unvisit = list([x for x in range(num_city) if x != self.start_node and x != self.end_node])
            current = self.start_node
            j = 1
            while len(unvisit) != 0:
                P = []
                for v in unvisit:
                    P.append(self.Tau[current][v] ** self.alpha * self.Eta[current][v] ** self.beta)
                P_sum = sum(P)
                P = [x / P_sum for x in P]
                index = self.rand_choose(P)
                current = unvisit[index]
                self.Table[i][j] = current
                unvisit.remove(current)
                j += 1
            self.Table[i][j] = self.end_node  # 添加固定终点


    # 计算不同城市之间的距离
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


    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    
    # 更新信息素
    def update_Tau(self):
        delta_tau = np.zeros([self.num_city, self.num_city])
        paths = self.compute_paths(self.Table)
        for i in range(self.m):
            for j in range(self.num_city - 1):
                a = self.Table[i][j]
                b = self.Table[i][j + 1]
                delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
            a = self.Table[i][0]
            b = self.Table[i][-1]
            delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
        self.Tau = (1 - self.rho) * self.Tau + delta_tau

    def aco(self):
        best_lenth = math.inf
        best_path = None
        for cnt in range(self.iter_max):
            # 生成新的蚁群
            self.get_ants(self.num_city)  # out>>self.Table
            self.paths = self.compute_paths(self.Table)
            # 取该蚁群的最优解
            tmp_lenth = min(self.paths)
            tmp_path = self.Table[self.paths.index(tmp_lenth)]
            # 可视化初始的路径
            if cnt == 0:
                init_show = self.location[tmp_path]
                init_show = np.vstack([init_show, init_show[0]])
            # 更新最优解
            if tmp_lenth < best_lenth:
                best_lenth = tmp_lenth
                best_path = tmp_path
            # 更新信息素
            self.update_Tau()

            # 保存结果
            self.iter_x.append(cnt)
            self.iter_y.append(best_lenth)
            print(cnt,best_lenth)
        return best_lenth, best_path

    def run(self):
        best_length, best_path = self.aco()
        return self.location[best_path], best_length


# 读取数据,包括巡检点和分类结果
def read_clustered_data(coords_path, cluster_path):
    """
    读取坐标和聚类结果文件，将巡检点按簇划分。
    
    返回:
        clustered_data (dict): key为簇ID，value为该簇对应的 numpy 坐标数组。
    """
    # 1. 读取坐标数据
    coords = []
    with open(coords_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:  # 确保不是空行
                coords.append([float(x) for x in parts])
    coords = np.array(coords)
    
    # 2. 读取聚类标签
    clusters = []
    with open(cluster_path, 'r') as f:
        for line in f:
            val = line.strip()
            if val:
                clusters.append(int(val))
    clusters = np.array(clusters)
    
    # 3. 按簇分离数据
    unique_clusters = np.unique(clusters)
    clustered_data = {}
    for cid in unique_clusters:
        # 获取该簇对应的所有坐标
        clustered_data[cid] = coords[clusters == cid]
        
    return clustered_data

if __name__ == '__main__':
    # 文件路径配置
    User_num = 40
    Cluster_num = 4
    
    # 文件路径
    coords_file = '../results/datas/Users_%d.txt' % User_num
    # cluster_file = 'results/datas/cluster/Users_%d_ClusteredUAV_%d.txt' % (User_num, Cluster_num)
    cluster_file = '../results/datas/cluster/Users_%d_Clustered_comm_4DUAV_%d.txt' % (User_num, Cluster_num)
    # 固定的起点和终点坐标 (这里参考了你之前Train脚本里的数据，可自行修改)
    #uav = 2
    # start_loc = np.array([14.76, 14.83, 0.0]) 
    # end_loc = np.array([27.62, 23.47, 0.0])
    
    #uav = 3
    # start_loc = np.array([32.88, 22.67, 0.0])
    # end_loc = np.array([21.62, 48.47, 0.0])
    
    #uav = 4
    start_loc = np.array([34.12, 28.79, 0.0])
    end_loc = np.array([38.46, 45.23, 0.0])
    
    # 读取并分组数据
    clustered_data = read_clustered_data(coords_file, cluster_file)
    
    # 初始化绘图画布，显式设置白色背景
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    colors = ['blue', 'green', 'purple', 'brown'] # 不同无人机路径使用不同颜色
    
    # 遍历每个簇，独立进行路径规划
    for cid, data in clustered_data.items():
        print(f"\n--- 正在为 簇(UAV) {cid} 规划路径 ---")
        
        # 将起点(插在最前)和终点(插在最后)与当前簇的任务点合并
        # 此时索引 0 为起点，索引 -1 (或 shape[0]-1) 为终点
        full_data = np.vstack([start_loc, data, end_loc])
        num_nodes = full_data.shape[0]
        
        # 初始化并运行 ACO
        aco = ACO(num_city=num_nodes, data=full_data.copy(), start_node=0, end_node=num_nodes-1)
        Best_path, Best_length = aco.run()
        
        print(f"簇 {cid} 规划完成，最佳路径长度: {Best_length:.2f}")
        
        # 绘制当前无人机的路径
        color = colors[cid % len(colors)]
        # 画任务点
        plt.scatter(Best_path[1:-1, 0], Best_path[1:-1, 1], c=color, marker='^', s=50, label=f'UAV {cid} Targets')
        # 画连线
        plt.plot(Best_path[:, 0], Best_path[:, 1], '-', c=color, linewidth=2, label=f'UAV {cid} Path')

    # 统一绘制起点和终点
    plt.scatter(start_loc[0], start_loc[1], color='orange', s=150, marker='o', edgecolors='black', label='Start', zorder=5)
    plt.scatter(end_loc[0], end_loc[1], color='red', s=150, marker='*', edgecolors='black', label='End', zorder=5)
    
    # 设置图表细节
    plt.xlabel('x (meter)', fontsize=14)    
    plt.ylabel('y (meter)', fontsize=14)
    plt.title('ACO Paths for Clustered UAVs', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 确保保存的图片没有透明背景，全为白底
    if not os.path.exists('../results/figs/'):
        os.makedirs('../results/figs/')
    plt.savefig('../results/figs/sequence/ACO_Clustered_Paths_UAV_%d.png' % Cluster_num, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()