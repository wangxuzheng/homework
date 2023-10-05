"""
Sklearn中的make_blobs方法生成数据，用K-Means聚类并可视化。

"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans
from clustering_performance import clusteringMetrics


def Kmeans3():
    point_coordinate, point_type,centers = make_blobs(n_samples=400, centers=3,return_centers=True)

    #设置颜色
    z = np.array(point_type)
    colors = np.array(["red","green","blue"])

    # 第一幅图
    plt.subplot(1, 2, 1)
    plt.scatter(point_coordinate[:, 0], point_coordinate[:, 1], s=100, marker="o", edgecolors='black', c=colors[z])
    plt.title('data by make_moons()')

    # 第二幅图
    plt.subplot(1, 2, 2)
    plt.title('K-means(K=3)')
    cluster = KMeans(n_clusters=3, random_state=9).fit(point_coordinate)
    centroid = cluster.cluster_centers_  # 查看聚类后的质心
    y_pred = cluster.labels_  # 获取训练后对象的每个样本的标签

    y = np.array(y_pred)
    plt.scatter(point_coordinate[:, 0], point_coordinate[:, 1], s=100, marker="o", edgecolors='black', c=colors[y])
    plt.scatter(centroid[:, 0], centroid[:, 1], s=100, marker="*", edgecolors='black', c='black')
    acc, nmi, ari = clusteringMetrics(point_type, y_pred)
    plt.suptitle('ACC = {}, NMI = {}, ARI = {}'.format(acc, nmi, ari))
    plt.show()


