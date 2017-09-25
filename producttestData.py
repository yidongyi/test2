#-*- coding:utf-8-*
import numpy as np;
import pandas as pd;
data = np.random.random([100,10]);
print(data)

data_excel=pd.DataFrame(data,columns=['a','b','c','d','e','f','g','h','i','j']);
print(data_excel)
data_excel.to_excel('data_excel.xlsx',sheet_name='one')
