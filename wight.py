#-*- coding:utf-8-*
import pandas as pd

def import_data_iris(file):
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
    return data
if __name__=='__main__':
  import numpy as np
  data = import_data_iris('iris.txt')
  print('原始数据：\n',data)
  #数据处理 0-均值规范化
  import numpy as np
  data = np.mat(data)
  data =np.multiply((data-data.min()),1/(data.max()-data.min()))
  print('数据处理 0-均值规范化:\n',np.mat(data))

  minmat=np.full((data.shape[0],data.shape[1]),np.mat(data).min(0))
  data=(data-minmat)/(data.max(0)-data.min(0))
  print('数据标准化：',data)

  CV=data.std(0)/data.mean(0)


  W=np.multiply(CV,1/np.sum(CV))

  B=np.full((W.shape[1],W.shape[1]),W)
  print('B-----',B)

  C=np.full((W.shape[1],W.shape[1]),W.T)
  print('C',C)

  A=np.multiply(B,1/C)

  print('A--',A)
