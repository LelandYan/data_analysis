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

#
# unames = ['user_id','gender','age','occupation','zip']
# users = pd.read_table('users.dat',sep='::',header=None,names=unames,engine='python')
# rnames = ['user_id','movie_id','rating','timestamp']
# ratings = pd.read_table('ratings.dat',sep="::",header=None,names=rnames,engine='python')
# mnames = ['movie_id','title','genres']
# movies = pd.read_table('movies.dat',sep="::",header=None,names=mnames,engine='python')
# # print(users[:5])
# data = pd.merge(pd.merge(ratings,users),movies)
# mean_ratings = data.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
# raings_by_title = data.groupby('title').size()
# active_titles = raings_by_title.index[raings_by_title >= 250]
# mean_ratings = mean_ratings.ix[active_titles]
# top_female_ratings = mean_ratings.sort_index(by='F',ascending=False)
# mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
# sorted_by_diff = mean_ratings.sort_index(by='diff')
# #print(sorted_by_diff[::-1][:15])
# rating_std_by_title = data.groupby('title')['rating'].std()
# rating_std_by_title = rating_std_by_title.ix[active_titles]
# print(rating_std_by_title.sort_values(ascending=False)[:10])

names1880 = pd.read_csv('yob1880.txt',names=['name','sex','births'])
#print(names1880)
#print(names1880.groupby('sex').births.sum())

years = range(1880,2011)
pieces = []
columns = ['name','sex','births']
for year in years:
    path = f'babynames/yob{year}.txt'
    frame = pd.read_csv(path,names=columns)

    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces,ignore_index=True)
# print(names)
total_births = names.pivot_table('births',index='year',columns='sex',aggfunc=sum)
#print(total_births[-5:])
total_births.plot(title='Total births by set and year')
# plt.show()


def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group
names = names.groupby(['year','sex']).apply(add_prop)

#print(np.allclose(names.groupby(['year','sex']).prop.sum(),1))
def get_top1000(group):
    return group.sort_index(by='births',ascending=False)[:1000]
grouped = names.groupby(['year','sex'])
top1000 = grouped.apply(get_top1000)
#print(top1000)
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births',index='year',columns='name',aggfunc=sum)
subset = total_births[['John','Harry','Mary','Marilyn']]
subset.plot(subplots=True,figsize=(12,10),grid=False,title='Number of births per year')
table = top1000.pivot_table('prop',index='year',columns='sex',aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))
df = boys[boys.year == 2010]
prop_cumsum = df.sort_index(by='prop',ascending=False).prop.cumsum()
#print(prop_cumsum.searchsorted(0.5))
df = boys[boys.year == 1900]
in1900 = df.sort_index(by='prop',ascending=False).prop.cumsum()
#print(in1900.searchsorted(0.5)+1)
def get_quantile_count(group,q=0.5):
    groups = group.sort_index(by='prop',ascending=False)
    return groups.prop.cumsum().searchsorted(q)+1
diversity = top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
# print(diversity.head())

get_last_letter = lambda x : x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = "last_letter"
table = names.pivot_table('births',index=last_letters,columns=['sex','year'],aggfunc=sum)

subtable = table.reindex(columns=[1910,1960,2010],level='year')
print(subtable.head())