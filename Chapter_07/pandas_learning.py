import pandas as pd
import numpy as np
from pandas import DataFrame, Series

# 1
# 可以根据一个或多个键将不同的DateFrame中的行连接起来，
# pd.merge()
# 可以沿着一条轴将多个对象堆叠到一起
# pd.concat()
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
# print(df1)
# print(df2)
# print(pd.merge(df1, df2))
# print(pd.merge(df1, df2, on='key'))
df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
# 默认情况下，merge()做的是'inner'连接，就是结果中的键的交集，其他的方式还有，left，right，outer
# print(pd.merge(df3, df4, left_on='lkey', right_on='rkey'))
# print(pd.merge(df3, df4, how='outer'))
df5 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df6 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})
# print(pd.merge(df5, df6, on='key', how='left'))
left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
# print(pd.merge(left, right, on=['key1', 'key2'], how='outer'))
# merger有一个更实用的suffixes选项，用于指定附加到左右两个DataFrame对象的重叠列名上的字符串
# print(pd.merge(left, right, on='key1', suffixes=('_left', '_right')))

# 2 索引上的合并
# left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
# right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
# print(pd.merge(left1, right1, left_on='key', right_index=True, how='outer'))

# 轴向连接 concatenation(连接) binding(绑定) stacking(堆叠)
# arr = np.arange(12).reshape((3, 4))
# print(np.concatenate([arr, arr], axis=1))
s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
# print(pd.concat([s1, s2, s3],axis=1))
s4 = pd.concat([s1 * 5, s3])
# print(s4)
result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])
# print(result)
# print(result.unstack())
# 合并重叠数据
a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan])
b = Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
# print(a)
# print(b)
# 一种用于表达一种矢量化的if-else
# print(np.where(pd.isnull(a), b, a))
# print(b[:-2].combine_first(a[2:]))
# 重塑层次化索引
# stack():将数据的列旋转为行
# unstack():将数据的行转为列
s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
# print(data2.unstack())
# stack默认会滤去缺失数据，因此该运算是可逆的
# pivot用于时间序列

# 数据转化
# 去除重复的数据
# duplicated()方法返回一个布尔型Series,表示各行是否是重复行
# drop_duplicates()删除重复的行

