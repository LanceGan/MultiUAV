import random
import math
import numpy as np
import matplotlib.pyplot as plt
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
        # fruits中存每一个个体是下标的list
        self.dis_mat = self.compute_dis_mat(num_city, data)
        #print("self.dis_mat",self.dis_mat)
        #print("debugu")
        self.fruits = self.greedy_init(self.dis_mat,num_total,num_city)
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]
        init_best = self.location[init_best]

        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index  = 1 # 起点
        # second_index = 0 # 第二个点
        end_index = num_city-1
        result = []
        label = True
        for i in range(num_total):
            rest = [x for x in range(1, num_city-1)] # 待选点[1,2,3,4,5,6] 去除起点和终点
            # 所有起始点都已经生成了
            if start_index >= num_city-1:
                start_index = np.random.randint(1, num_city-2)
                # print("start_index",start_index)
                result.append(result[start_index].copy())
                continue
            current = start_index
            # print(current)
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [0,current] # 添加起点和第一个点的可行路径
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest: # 从待选点中选择路径最短的点
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                # 判断是否有可行的路径点
                if(tmp_choose != -1): #
                    label = True
                    current = tmp_choose  # 更新下一个路径点
                    rest.remove(tmp_choose)  # 从代选点移除可行点
                    result_one.append(tmp_choose)  # 将可行点加入集合
                else:
                    label = False
                    # print("running to this")
                    break

            if(label):
                # print("runing in label")
                result_one.append(end_index)
                result.append(result_one)
            # print("runing in label")
            start_index += 1
        return result
    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if (i == j):
                    dis_mat[i][j] = np.inf
                    continue
                # 起点不能直接到达终点
                if(i==0 and j==num_city-1):
                    dis_mat[i][j] = np.inf
                    continue
                # 所有城市不能到达起点
                if(i!=0 and j==0):
                    dis_mat[i][j] = np.inf
                    continue
                # 终点不能到达任何一个点
                if(i==num_city-1):
                    dis_mat[i][j] = np.inf
                    continue

                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))

                # tmp = path_Dji.compt_dis_Dji(a * 10, b * 10)  # 注意数据的单位
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat,goback=False):
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()


        result = 0.0

        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ga_cross(self, x, y):
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)

        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
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
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        # order =
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.num_total:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.fruits = fruits

        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in tqdm(range(1, self.iteration + 1)):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1./best_score)
            #print(i,1./best_score)
        #print(1./best_score)
        return self.location[BEST_LIST], 1. / best_score


# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data

def get_path_len(path):
    """ 计算路径的长度 """
    path_length = 0
    # print(path)
    for i in range(1, len(path)):
        node1_x = path[i][0]
        node1_y = path[i][1]
        node2_x = path[i - 1][0]
        node2_y = path[i - 1][1]
        path_length += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
    return path_length

if __name__ == "__main__":
    data = read_tsp('results/datas/Users_7.tsp')

    data = np.array(data)
    data = data[:, 1:]
    Best, Best_path = math.inf, None
    print(data.shape[0])
    model = GA(num_city=data.shape[0], num_total=25, iteration=500, data=data.copy())
    print("=======================loading========================")
    path, path_len = model.run()
    if path_len < Best:
        Best = path_len
        Best_path = path
    # np.savez("GA_mindis_path",Best_path)

    path_len_cmp  = get_path_len(path)
    print("遍历顺序",path,"路径长度:",path_len,path_len_cmp)


    npzfile_sinr = np.load('results/datas/Radio_datas_A2G.npz')
    OutageMapActual = npzfile_sinr['arr_0'] 
    Y_vec2 = npzfile_sinr['arr_2']  # [0,1....100]标号
    X_vec2 = npzfile_sinr['arr_3']
    norm = plt.Normalize(vmin=-5, vmax=20)
    plt.contourf(np.array(Y_vec2) * 10, np.array(X_vec2) * 10, 1-OutageMapActual)
    v = np.linspace(-10, 30, 6, endpoint=True)
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
    plt.scatter(Best_path[0, 0]*1000, Best_path[0, 1]*1000, color='orange') # 画出起点
    plt.scatter(Best_path[-1, 0]*1000, Best_path[-1, 1]*1000, color='blue') # 画出终点
    plt.scatter(Best_path[1:-1, 0]*1000, Best_path[1:-1, 1]*1000, c='red', marker='^') # 画出任务点
    plt.plot(Best_path[:, 0]*1000, Best_path[:, 1]*1000,'b-')
    plt.xlabel('x (meter)', fontsize=14)
    plt.ylabel('y (meter)', fontsize=14)
    plt.title('GA Path', fontsize=16)
    plt.savefig('results/figs/GA_path.png', dpi=300, bbox_inches='tight')
    plt.show()