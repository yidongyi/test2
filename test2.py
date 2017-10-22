import matplotlib.pyplot as plt;
import numpy as np;
import pandas as pd;


# x=range(100)
# y=np.random.random([100,1])*10
# print(y)
# plt.xlim(0,300)
# plt.ylim(0,13)
# plt.plot(x ,y,'*')
# plt.show()
# a = np.mat([1,2,3,4,5])
#
# print(a)
# b = np.where(a==5)
# print(b[1])
#
# c=[[1,2,3],[4,5,6],[7,8,9]]
# print(len(c))
# print(min(c[0]))
# print(max(c[1]))


data=pd.read_excel('test.xlsx')

print (data.as_matrix().max(0))
print (data.as_matrix().min(0))
minmat=np.full((3,7),data.as_matrix().min(0))
data=(data-minmat)/(data.as_matrix().max(0)-data.as_matrix().min(0))
print(data)
print(data.as_matrix().std(0))
print(data.as_matrix().mean(0))
