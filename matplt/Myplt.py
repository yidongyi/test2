import  matplotlib.pyplot as plt;
import numpy as np
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']#正常显示中文
plt.rcParams['axes.unicode_minus']=False;#正常显示负号


'''
plot()
'''
figure1=plt.figure('plot',figsize=(7,5));
x=np.linspace(0,2*np.pi,50);
y = np.sin(x);
plt.plot(x,y,'bp--');
plt.xlabel('中文')
plt.ylabel('abc')
figure1.show()

'''
pie()
'''
figure2 = plt.figure('pie')
labels = 'Frogs' , 'Hogs', 'Dogs', 'Logs'
sizes = [15,30,45,10]
colors = ['yellowgreen','gold','lightskyblue','lightcoral']
explode = (0,0.1,0,0)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90)
plt.axis('equal')
figure2.show()

'''
hist()
'''
figure3 = plt.figure('hist')
x= np.random.randn(1000)
plt.hist(x,10)
figure3.show()

'''
boxplot()
'''

x = np.random.randn(1000)
D = pd.DataFrame([x,x+1]).T
D.plot(kind = 'box')
plt.show()

'''
plot(logx=True)
'''
figure5 = plt.figure('plot(log)')
x = pd.Series(np.exp(np.arange(20)))
x.plot(label='原始数据',legend = True)
plt.show()
x.plot(logy=True,label='对数数据',legend = True)
plt.show()
