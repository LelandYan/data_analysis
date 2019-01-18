# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/18 11:05'
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# # 设置字符集，防止中文乱码
# mpl.rcParams['font.sans-serif'] = ['simHei']
# mpl.rcParams['axes.unicode_minus'] = False
#
# # 定义目标函数
# def l_model(x):
#     params = np.arange(1,x.shape[-1]+3)
#     y = np.sum(params[:-2] * x) + np.random.randn(1) * 0.1 + 5 * params[-2] * x[0] * x[1] + 5 * params[-1] * x[1] * x[2]
#     return y
#
# # 定义数据集
# x = pd.DataFrame(np.random.rand(500,6))
# y = x.apply(lambda  x_rows:pd.Series(l_model(x_rows)),axis=1)
#
# # 划分训练集和测试集
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2)
#
#
# # 定义管道,在models中可以定义多个Pipeline
# models = [
#     Pipeline(memory=None,
#              steps=[
#                  ('StandardScaler',StandardScaler()),# 数据标准化
#                  ('Poly',PolynomialFeatures()),# 多项式扩展
#                  ('LinearRegression',LinearRegression()),# 线性回归
#              ])
# ]
# model = models[0]
#
# t = np.arange(len(x_test))
# N = 4
# scale_pool = [True,False]
# degree_pool = np.arange(1,N,1)
# regressor_pool = [True,False]
# gsize = len(scale_pool)*len(degree_pool)*len(regressor_pool)
#
# # 管道参数遍历训练模型
# line_width = 3
# plt.figure(figsize=(12,5),facecolor='w')# 设置绘图窗口的大小，颜色
# ical = 1
# for i,s in enumerate(scale_pool):
#     for j,d in enumerate(degree_pool):
#         for k,r in enumerate(regressor_pool):
#             plt.subplot(gsize,1,ical)
#             plt.plot(t,y_test,'r-',label='真实值')
#             # 设置管道参数
#             model.set_params(StandardScaler__with_mean=s) # 标准化时候是否中心化
#             model.set_params(Poly__degree=d) # 多项式扩展项的阶数
#             model.set_params(LinearRegression__fit_intercept=r)  # 回归的时候是否考虑截距
#
#             # cress_validation训练
#             cv_results = cross_validate(model,x_train,y_train,cv=5,return_train_score=False)
#             # 输出cross_validation结果
#             print(cv_results['test_score'])
#             # 转化为95%的置信区间
#             scores = cv_results['test_score']
#             print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#             ical += 1


# iris = datasets.load_iris()
# # print(iris.data.shape,iris.target.shape)
# X_train,X_test,y_train,y_test = cross_validate(iris.data,iris.target)
# print(X_train.shape)
import os
def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []
    labelMat = []
    for eachline in fr:
        lineArr = []
        curLine = eachline.strip().split('\t')
        for i in range(3,len(curLine)-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(int(curLine[-1]))
    fr.close()
    return dataMat,labelMat

def splitDataSet(fileName,split_size,outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fr = open(fileName,'r')
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)
    arr = np.arange(num_line)
    np.random.shuffle(arr)
    list_all = arr.tolist()
    each_size = (num_line+1)/split_size
    split_all = []
    each_split =[]
    count_num = 0
    count_split = 0
    for i in range(len(list_all)):
        each_split.append(onefile[int(list_all[i])].strip())
        count_num += 1
        if count_num == each_size:
            count_split += 1
            array_ = np.array(each_split)
            np.savetxt(outdir+"/split_"+str(count_split)+'.txt',array_,fmt="%s",delimiter="\t")




















