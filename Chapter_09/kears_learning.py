import pandas as pd
import numpy as np
from pandas import DataFrame, Series

# GroupBy
df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1': np.random.rand(5),
                'data2': np.random.randn(5)})
grouped = df['data1'].groupby(df['key1'])
# print(grouped.mean())
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
# print(means.unstack())
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
# GroupBy的size方法,他可以返回一个含有分组的大小的Series

# 对分组进行迭代
# for name, group in df.groupby('key1'):
#     print(name, group)

# for (k1, k2), group in df.groupby(['key1', 'key2']):
#     print(k1, k2, group)


people = DataFrame(np.random.randn(5, 5), columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
# print(people)
people.ix[2:3, ['b', 'c']] = np.nan
# print(people)
mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f': 'orange'}
by_column = people.groupby(mapping, axis=1)
# print(by_column.sum())

# 通过函数进行分组
people.groupby(len).sum()

key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()

# 根据索引级别分组
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                     [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
hier_df.groupby(level='cty', axis=1).count()
# print(hier_df.groupby(level='cty',axis=1).count())

# 数据聚合
# grouped = df.groupby('key1')
# print(grouped['data1'])
# print(grouped['data1'].quantile(0.9))

tips = pd.read_csv('tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
# print(tips[:6])

# 面向列的多函数应用
# grouped = tips.groupby(['sex', 'smoker'])
# grouped_pct = grouped['tip_pct']
# grouped_pct.agg('mean')

# 聚合数据都有唯一的分组键组成的索引，可以通过向groupby中传入as_index=False
# print(tips.groupby(['sex','smoker'],as_index=False).mean())
k1_means = df.groupby('key1').mean().add_prefix('mean_')
pd.merge(df, k1_means, left_on='key1', right_index=True)
# print(pd.merge(df,k1_means,left_on='key1',right_index=True))
key = ['one', 'two', 'one', 'two', 'one']
print(people.groupby(key).transform(np.mean))
