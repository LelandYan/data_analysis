# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/20 20:14'

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
# 逻辑回归
from sklearn.linear_model import LogisticRegression
# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
# k-近邻
from sklearn.neighbors import KNeighborsClassifier
# 决策树
from sklearn.tree import DecisionTreeClassifier
# svm
from sklearn import svm

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]
def show_accuracy(y_hat,y_test,param):
    pass
path = 'iris.data'

data = np.loadtxt(path,dtype=float,delimiter=',',converters={4:iris_type})
x,y = np.split(data,(4,),axis=1)
x = x[:,:2]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.6)

# 训练svm分类器
clf = svm.SVC(C=0.8,kernel='rbf',gamma=20,decision_function_shape='ovr')
clf.fit(x_train,y_train.ravel())

# 精度
print(clf.score(x_train,y_train))
# y_hat = clf.predict(x_train)
print(clf.score(x_test,y_test))
# y_hat = clf.predict(x_test)


x1_min,x1_max = x[:,0].min(),x[:,0].max()
x2_min,x2_max = x[:,1].min(),x[:,1].max()
x1,x2 = np.mgrid[x1_min:x1_max:200j,x2_min:x2_max:200j]
grid_test = np.stack((x1.flat,x2.flat),axis=1)

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

# print 'grid_test = \n', grid_test
grid_hat = clf.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

alpha = 0.5
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
# plt.grid()
plt.show()