# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/18 20:24'

import pandas as pd
from pandas import DataFrame,Series

obj = Series([4,7,-5,3])
#print(obj)
obj2 = Series([4,7,-5,3],index=['b','d','a','c'])
#print(obj2)
sdata = {'Ohio':3500,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3 = Series(sdata)
print(obj3)