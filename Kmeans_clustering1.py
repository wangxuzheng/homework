"""
Sklearn中的make_circles方法生成数据，用K-Means聚类并可视化。
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score


def Kmeans():
    point_coordinate, point_type = make_circles(n_samples=400, noise=0.1, factor=0.1)

    #设置颜色
    z = np.array(point_type)
    colors = np.array(["red","green"])

    # 第一幅图
    plt.subplot(1, 2, 1)
    plt.scatter(point_coordinate[:, 0], point_coordinate[:, 1], s=100, marker="o", edgecolors='black', c=colors[z])
    plt.title('data by make_circles()')

    # 第二幅图
    plt.subplot(1, 2, 2)
    plt.title('K-means(K=2)')
    cluster = KMeans(n_clusters=2,random_state=9).fit(point_coordinate)
    centroid = cluster.cluster_centers_ #查看聚类后的质心
    y_pred = cluster.labels_ #获取训练后对象的每个样本的标签
    y = np.array(y_pred)
    colors = np.array(["red", "green"])
    plt.scatter(point_coordinate[:, 0], point_coordinate[:, 1], s=100, marker="o", edgecolors='black',c=colors[y])
    plt.scatter(centroid[:,0],centroid[:,1],s=100, marker="*", edgecolors='black',c = 'blue')
    acc = accuracy_score(point_type,y_pred)
    nmi = normalized_mutual_info_score(point_type,y_pred)
    ari = adjusted_rand_score(point_type,y_pred)
    plt.suptitle('ACC = {}, NMI = {}, ARI = {}'.format(acc,nmi,ari))
    plt.show()


