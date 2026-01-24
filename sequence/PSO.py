import random
import math
import numpy as np
import matplotlib.pyplot as plt

class PSO(object):
    def __init__(self, num_city, data):
        self.iter_max = 500  # 迭代数目
        self.num = 200  # 粒子数目
        self.num_city = num_city  # 城市数
        self.location = data # 城市的位置坐标
        # 计算距离矩阵
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵
        # 初始化所有粒子
        # self.particals = self.random_init(self.num, num_city)
        self.particals = self.greedy_init(self.dis_mat,num_total=self.num,num_city =num_city)
        self.lenths = self.compute_paths(self.particals)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        # 画出初始的路径图
        init_show = self.location[init_path]
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 记录当前的全局最优解,长度是iteration
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

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        # 更新当前的全局最优
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
        # 更新当前的个体最优
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    def cross(self, cur, best):
        one = cur.copy()
        # 固定起点终点
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


    # 迭代操作
    def pso(self):
        for cnt in range(1, self.iter_max):
            # 更新粒子群
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                # 与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l
                # 变异
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                # 更新该粒子
                self.particals[i] = one
                self.lenths[i] = tmp_l
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particals()
            # 更新输出解
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            #print(cnt, self.best_l)
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        # 画出最终路径
        return self.location[best_path], best_length

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
        

def pso_on_clusters(points: np.ndarray, labels: np.ndarray, save_path=None,ini_loc=None, end_loc=None):
    res = {}
    uniq = np.unique(labels)
    for c in uniq:
        idxs = np.where(labels == c)[0]        # 原始索引
        if len(idxs) <= 2:
            res[int(c)] = idxs.tolist()
            continue
        locs = points[idxs]                    # 子集坐标 (shape: n_c x 2)
        # 构建一个映射数组，将 locs 中每一行映射回原始索引，
        # 对于插入的 ini_loc/end_loc 使用 -1 作为占位符
        map_idxs = np.array(idxs, dtype=int)
        # 每一个簇都是要从起点出发并回到终点（可选插入）
        if ini_loc is not None and end_loc is not None:
            locs = np.vstack([ini_loc, locs, end_loc])
            map_idxs = np.concatenate((np.array([-1], dtype=int), map_idxs, np.array([-1], dtype=int)))
        elif ini_loc is not None:
            locs = np.vstack([ini_loc, locs])
            map_idxs = np.concatenate((np.array([-1], dtype=int), map_idxs))
        elif end_loc is not None:
            locs = np.vstack([locs, end_loc])
            map_idxs = np.concatenate((map_idxs, np.array([-1], dtype=int)))
        # 注意：当前 sequence/PSO.py 中的 `PSO` 将第 0 个和最后一个点视为固定起点/终点
        # 所以这里 locs[0] 和 locs[-1] 会被当作固定端点；可以按需选择端点顺序
        pso = PSO(len(locs), locs)
        best_l, best_path = pso.pso()          # 或用 pso.run()
        best_path = np.array(best_path)        # 索引相对于 locs
        # 使用 map_idxs 将 locs 的索引映射回原始数据索引，过滤掉插入的起/终点(-1)
        mapped = map_idxs[best_path]
        # 保留原始点索引，去掉-1占位
        orig_order = mapped[mapped != -1]
        res[int(c)] = orig_order.tolist()
    if save_path is not None:
        np.savez(save_path, result=res)
    return res

if __name__ == '__main__':
    User_num = 20
    Cluster_num = 2
    ini_loc = [[14.76, 14.83],[32.88, 22.67],[34.12, 28.79]][0]
    end_loc = [[27.62, 23.47],[21.62, 48.47],[38.46, 45.23]][0]
    
    Cluster_result = 'results/datas/cluster/Users_%d_Clustered'% User_num+'UAV_'+str(Cluster_num)+'.txt'
    Cluster_label = load_cluster(Cluster_result,User_num)

    Users_path = 'results/datas/Users_%d.txt' % User_num
    x0_user, y0_user = load_user(Users_path,User_num)
    points = np.column_stack((x0_user, y0_user))
    
    save_path = 'results/datas/sequence/Users_%d_Clusteredsave_path_Path'% User_num+'UAV_'+str(Cluster_num)
    res = pso_on_clusters(points, np.array(Cluster_label),save_path,ini_loc,end_loc)
    print("最终路径结果（按簇划分）:", res)

    


