# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/17 21:34'

import json

path = 'example.txt'

# with open(path) as f:
#     print(f.readline())

records = [json.loads(line) for line in open(path)]
# print(records[0])
# print(records[0]['tz'])
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# print(time_zones[:10])

def get_count(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

counts = get_count(time_zones)

from collections import defaultdict


def get_count2(sequence):
    # 所有的值均会初始化为 0
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts

def top_counts(count_dict,n=10):
    value_key_pairs = [(count,tz) for tz,count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

# print(top_counts(counts))

# from collections import Counter
# counts = Counter(time_zones)
#print(counts.most_common(10))


from pandas import DataFrame,Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

frame = DataFrame(records)
#print(frame['tz'].value_counts())
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unkown'
tz_counts = clean_tz.value_counts()
# print(tz_counts[:10])


# tz_counts[:10].plot(kind='barh',rot=0)
# plt.show()
# times = [counts[0] for counts in tz_counts]
# site = [counts[1] for counts in tz_counts]

result = Series([x.split()[0] for x in frame.a.dropna()])
#print(result.value_counts()[:8])
cframe = frame[frame.a.notnull()]
operaing_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
# print(operaing_system[:5])
by_tz_os = cframe.groupby(['tz',operaing_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
#print(by_tz_os.size().unstack())
#print(agg_counts[:10])
indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]
# print(count_subset)
count_subset.plot(kind='barh',stacked=True)
# plt.show()
normed_subset = count_subset.div(count_subset.sum(1),axis=0)
normed_subset.plot(kind='barh',stacked=True)
# plt.show()


unames = ['user_id','gender','age','occupation','zip']
users = pd.read_table('users.dat',sep='::',header=None,names=unames,engine='python')
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('ratings.dat',sep="::",header=None,names=rnames,engine='python')
mnames = ['movie_id','title','genres']
movies = pd.read_table('movies.dat',sep="::",header=None,names=mnames,engine='python')
print(users[:5])


