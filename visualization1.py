"""
生成数据 make_circles 和 make_moons，并显示 X = 400x2, Y = {0, 1}400 ，画图

"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons



def draw_cycle():
    point_coordinate, point_type = make_circles(n_samples=400, noise=0.1, factor=0.1)
    x = point_coordinate[:, 0]
    y = point_coordinate[:, 1]
    plt.figure("circle")
    plt.scatter(x, y, s=100, marker="o", edgecolors='black', c=point_type, cmap='Spectral')
    # c=point_type划分两种标签数据的颜色    s代表size数据点
    plt.title('data by make_circles()')
    plt.show()


def draw_moon():
    point_coordinate, point_type = make_moons(n_samples=400, noise=0.1)
    x = point_coordinate[:, 0]
    y = point_coordinate[:, 1]
    plt.figure("moon")
    plt.scatter(x, y, s=100, marker="o", edgecolors='black', c=point_type, cmap='Spectral')
    plt.title('data by make_moons()')
    plt.show()
