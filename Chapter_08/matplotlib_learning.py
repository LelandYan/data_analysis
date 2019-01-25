import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# plt.plot(randn(50).cumsum(), 'k--')
# _ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
# ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))

# 调整subplot周围的间距
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# for i in range(2):
#     for j in range(2):
#         axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.plot(randn(30).cumsum(), 'ko--')
# plt.plot(randn(30).cumsum(), color='k', linestyle='dashed', marker='o')
# data = randn(30).cumsum()
# # plt.plot(data, 'k--', label='Default')
# plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
# plt.legend(loc='best')
# plt.xlim()
# plt.savefig('figpath.svg')
# plt.show()
# from io import BytesIO
# buffer = BytesIO()
# plt.savefig(buffer)
# plot_data = buffer.getvalue()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(randn(1000).cumsum())
# plt.show()

# from datetime import datetime
# import pandas as pd
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# data = pd.read_csv('spx.csv', index_col=0, parse_dates=True)
# spx = data['SPX']
# spx.plot(ax=ax, style='k--',alpha=0.3)
# crisis_data = [
#     (datetime(2007, 10, 11), 'Peak of bull market'),
#     (datetime(2008, 3, 12), 'Bear Stearns Fails'),
#     (datetime(2008, 9, 15), 'Lehman Bankruptcy')
# ]
# for date, label in crisis_data:
#     ax.annotate(label, xy=(date, spx.asof(date) + 50),
#                 xytext=(date, spx.asof(date) + 200),
#                 arrowprops=dict(facecolor='black'),
#                 horizontalalignment='left', verticalalignment='top')
# ax.set_xlim(['1/1/2007', '1/1/2011'])
# ax.set_ylim([600, 1800])
# ax.set_title("Important dates in 2008-2009 financial crisis")
# plt.show()
# ax.savefig('figpath.svg')
# matplotlib配置
# plt.rc('figure', figsize=(10, 10))

from pandas import DataFrame, Series

# pandas中的绘图函数
# 线型图
# s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
# s.plot()
# plt.show()
# df = DataFrame(np.random.randn(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
# df.plot()
# plt.show()

# 柱状图 kind='bar/barh' Serise和DataFrame的索引将会被X，Y刻度
# fig, axes = plt.subplots(2, 1)
# data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
# data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)
# data.plot(kind='barh', ax=axes[1], color='k', alpha=0.7)
# plt.show()
import pandas as pd

# df = DataFrame(np.random.rand(6, 4),
#                index=['one', 'two', 'three', 'four', 'five', 'six'],
#                columns=pd.Index(['A', 'B', 'C', 'D'], names='Genus'))
# df.plot(kind='bar')
# df.plot(kind='barh', stacked=True, alpha=0.5)
# plt.show()
# tips = pd.read_csv('tips.csv')
# party_counts = pd.crosstab(tips.day,tips.size)
# print(party_counts.ix[:,2:5])

# 直方图和密度图
# tips = pd.read_csv('tips.csv')
# tips['tip_pct'] = tips['tip'] / tips['total_bill']
# tips['tip_pct'].hist(bins=20)
# tips['tip_pct'].plot(kind='kde')
# plt.show()

# comp1 = np.random.normal(0, 1, size=200)
# comp2 = np.random.normal(10, 2, size=200)
# values = Series(np.concatenate([comp1,comp2]))
# values.hist(bins=100,alpha=0.3,color='k',normed=True)
# values.plot(kind='kde',style='k--')
# plt.show()

# 散步图
# macro = pd.read_csv('macrodata.csv')
# # data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
# # # print(data[-5:])
# # trans_data = np.log(data).diff().dropna()
# # # print(trans_data[-5:])
# # plt.scatter(trans_data['m1'],trans_data['unemp'])
# # plt.title('Changes in log')
# # pd.scatter_matrix(trans_data,diagonal='kde',color='k',alpha=0.3)
# # plt.show()

# 绘制地图
data = pd.read_csv('Haiti.csv')
# 清除错误的信息
data = data[(data.LATITUDE > 18) & (data.LATITUDE < 20) & (data.LONGITUDE > -75) & (data.LONGITUDE < -70) & (
    data.CATEGORY.notnull())]


def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]


def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x) for x in cat_series))
    return sorted(set.union(*cat_sets))


def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split('|')[1]
    return code, names.strip()


print(get_english('2. Urgences logistiques | Vital Lines'))
