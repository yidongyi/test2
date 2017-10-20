#-*- coding:utf-8-*

def import_data_format_iris(file):
    """
    格式化数据，前四列为data，最后一列为cluster_location
    数据地址 http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
    """
    data = []
    cluster_location =[]
    with open(str(file), 'r') as f:
        for line in f:
            current = line.strip().split(",")
            current_dummy = []
            for j in range(0, len(current)-1):
                current_dummy.append(float(current[j]))
            j += 1
            if  current[j] == "Iris-setosa":
                cluster_location.append(0)
            elif current[j] == "Iris-versicolor":
                cluster_location.append(1)
            else:
                cluster_location.append(2)
            data.append(current_dummy)
    print ("加载数据完毕")
    return data , cluster_location

def k_fuzzy_means(data, cluster_number, m,w):
    """
    这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    参数是：簇数(cluster_number)和隶属度的因子(m)
    w 是属性权重
    """
    # 初始化隶属度矩阵U
    U = initialise_U(data, cluster_number)
    #print("-----初始化隶属度矩阵U------")
    #print_matrix(U)
    # 循环更新U
    while (True):
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]*w[i]
                    # 分母
                    dummy_sum_dum += (U[k][j] ** m)*w[i]
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)

        # 创建一个距离向量, 用于计算U矩阵。
        distance_matrix =[]
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)

        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += (distance_matrix[i][j ] / distance_matrix[i][k]) ** (2/(m-1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            print ("结束聚类_number",len(C))
            #计算有效性评价指数
            print('计算有效性评价指数V:',get_Vnew(U,C,data,m))
            break
    #print ("标准化 U")
    U = normalise_U(U)
    #print(U)
    return U
def K(x,y):
    return
