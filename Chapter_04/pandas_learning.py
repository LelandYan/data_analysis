# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/18 20:24'

import pandas as pd
import numpy as np
from pandas import DataFrame, Series

obj = Series([4, 7, -5, 3])
# print(obj)
obj2 = Series([4, 7, -5, 3], index=['b', 'd', 'a', 'c'])
# print(obj2)
sdata = {'Ohio': 3500, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
# print(data.drop(['Colorado','Utah']))
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
# print(s1+s2)
objs = Series([7, -5, 7, 4, 2, 0, 4])
# print(objs.rank())
datas = DataFrame({"Qu1":[1,3,4,3,4],
                  "Qu2":[2,3,1,2,3],
                  "Qu3":[1,5,2,4,4]})
result = datas.apply(pd.value_counts).fillna(0)
# print(result)

string_data = Series(['aardvark','artichoke',np.nan,'avocado'])
#print(string_data)
# print(string_data.isnull())

from numpy import nan as NA
data1 = Series([1,NA,3.5,NA,7])
# print(data1.dropna())
df = DataFrame(np.random.randn(7,3))
print(df)
df.ix[:4,1] = NA
df.ix[:2,2] = NA


ser = Series(np.arange(3),index=['a','b','c'])
print(ser[-1])

