# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/22 11:03'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pydotplus

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
    iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
    iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

    path = "iris.data"
    data = pd.read_csv(path,header=None)
    x = data[list(range(4))]
    y = LabelEncoder().fit_transform(data[4])
    x = x.iloc[:,:2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train,y_train)
    y_test_hat = model.predict(x_test)
    print('accuracy_score',accuracy_score(y_test,y_test_hat))

    # with open('iris.dot','w') as f:
    #     tree.export_graphviz(model,out_file=f)
    #
    # dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E[:2], class_names=iris_class,
    #                                 filled=True, rounded=True, special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('iris.pdf')
    # f = open('iris.png', 'wb')
    # f.write(graph.create_png())
    # f.close()

    y_test = y_test.reshape(-1)
    result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
    acc = np.mean(result)
    print(f'准确度: {100 * acc:f}')

    N, M = 50, 50  # 横纵各采样多少个值
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)  # 预测值
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x_test[0], x_test[1], c=y_test.ravel(), edgecolors='k', s=100, zorder=10, cmap=cm_dark,
                marker='*')  # 测试数据
    plt.scatter(x[0], x[1], c=y.ravel(), edgecolors='k', s=20, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('鸢尾花数据的决策树分类', fontsize=15)
    plt.show()