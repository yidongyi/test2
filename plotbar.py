# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False
X=[1,2,3,4,5,6,7,8,9,10]
Y=[ 7598 ,11748 ,11834 , 7465 ,6957  ,7263 , 7176 ,13250 ,12252 , 6153]
fig = plt.figure()
plt.bar(X,Y,0.4,color="green")
plt.xlabel("第几簇")
plt.ylabel("聚类个数")
plt.title("聚类结果")


plt.show()
