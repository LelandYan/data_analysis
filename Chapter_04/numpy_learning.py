# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/18 19:04'

import numpy as np
data1 = [6,7.5,8,0,1]
arr1 = np.array(data1)
#print(arr1)
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
#print(arr2)
#print(np.zeros((3,6)))
#print(np.empty((2,3,2)))
#print(np.arange(15))
data = np.random.randn(7,4)
#print(data)
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
print(arr)