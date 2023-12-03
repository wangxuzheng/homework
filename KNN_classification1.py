"""
Sklearn中的make_circles方法生成训练样本
并随机生成测试样本，用KNN分类并可视化。
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
import random
import numpy as np
from collections import Counter

def KNN():
    point_coordinate, point_type = make_circles(n_samples=400, noise=0.1, factor=0.1)

    k = 15
    # 随机生成测试样本
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    #设置颜色
    z = np.array(point_type)
    colors = np.array(["red","green"])

    # 第一幅图
    plt.subplot(1, 2, 1)
    plt.scatter(point_coordinate[:, 0], point_coordinate[:, 1], s=100, marker="o", edgecolors='black', c=colors[z])
    plt.title('data by make_circles()')
    plt.scatter(x, y, s=100, marker="*", edgecolors='black', c='blue')

    # 第二幅图
    plt.subplot(1, 2, 2)
    plt.scatter(point_coordinate[:, 0], point_coordinate[:, 1], s=100, marker="o", edgecolors='black', c=colors[z])
    plt.title('KNN(K=15)')

    # 转化为DataFrame格式
    data = {'x坐标': point_coordinate[:, 0], 'y坐标': point_coordinate[:, 1], }
    olddata = pd.DataFrame(data, dtype='double')

    # 计算欧式距离,距离排序
    new_x_y = [float(x), float(y)]
    distance = (((olddata - new_x_y) ** 2).sum(1)) ** 0.5  # 得到((x1-x2)^2+(y1-y2)^2)^0.5
    disdata = pd.DataFrame({'x坐标': point_coordinate[:, 0], 'y坐标': point_coordinate[:, 1], '距离': distance,'标签':point_type}, dtype='double').sort_values(
        by='距离')
    labels = disdata.iloc[:k, 3].tolist()
    # 使用Counter函数计算出现次数
    count_dict = Counter(labels)
    # 找到出现次数最多的元素
    most_common = count_dict.most_common(1) #[(0.0,30)]
    star_label = int(most_common[0][0]) #该随机点最大的可能性分类为K近邻点重复最多的类别。
    plt.scatter(x, y, s=100, marker="*", edgecolors='black', c=colors[star_label]) #按照分类来上色
    # 距离最短前k个
    plt.scatter(disdata.iloc[:k, 0], disdata.iloc[:k, 1], s=100, marker="o", edgecolors='black', c='blue')



    plt.show()


