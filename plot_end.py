#-*- coding:utf-8-*
import matplotlib.pyplot as plt;
import numpy as np;
import pandas as pd;
import xlrd


# plt.figure();
# plt.plot(c,'*')
# plt.xlim(-1,3)
# plt.ylim(-1,3)
# plt.show();

# data = pd.read_excel('data_excel.xlsx')
# print(data)
#
# data_mat=data.as_matrix(columns=None)


#print(data_mat)
#print(np.max(data_mat))
def plotend(data_mat):
   #  print(np.max(data_mat,1))#max of each row
    x=0;
    y=[]
    for b in range(data_mat.shape[0]):
        c = np.where(data_mat[b] == 1 )
        x=x+1
        y.append(c[1])
    plt.xlim(0,data_mat.shape[0])
    plt.ylim(-1,12)#聚类数
    plt.plot(range(x) ,list(y),'*')
    plt.xlabel('Number of samples')
    plt.ylabel(' cluster')
    plt.show()
