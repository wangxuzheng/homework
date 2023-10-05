"""
Sklearn中的make_circles方法生成训练样本
并随机生成测试样本，用KNN分类并可视化。
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
import random
import numpy as np

def KNN():
    point_coordinate, point_type = make_circles(n_samples=400, noise=0.1, factor=0.1)

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
    plt.scatter(x, y, s=100, marker="*", edgecolors='black', c='blue')

    # 转化为DataFrame格式
    data = {'x坐标': point_coordinate[:, 0], 'y坐标': point_coordinate[:, 1], }
    olddata = pd.DataFrame(data, dtype='double')

    # 计算欧式距离,距离排序
    new_x_y = [float(x), float(y)]
    distance = (((olddata - new_x_y) ** 2).sum(1)) ** 0.5  # 得到((x1-x2)^2+(y1-y2)^2)^0.5
    disdata = pd.DataFrame({'x坐标': point_coordinate[:, 0], 'y坐标': point_coordinate[:, 1], '距离': distance}, dtype='double').sort_values(
        by='距离')

    # 距离最短前k个
    k = 15
    plt.scatter(disdata.iloc[:k, 0], disdata.iloc[:k, 1], s=100, marker="o", edgecolors='black', c='blue')

    plt.show()


