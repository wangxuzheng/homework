"""
给定的图像数据集，探讨pca降维后特征个数与聚类性能的关系。

"""

import matplotlib.pyplot as plt
from utils import get_face_images
from sklearn.decomposition import PCA
from clustering_performance import clusteringMetrics
from sklearn.cluster import KMeans
import numpy as np
def eigenfaces_performance():
    ROW = 10  # 指定拼接图片的行数,即有多少人
    COL = 20  # 指定拼接图片的列数，即每个人有多少图片
    PATH = 'face_images/'
    n_components = 10  #用PCA降维，n_components为降到的维数   主成分数量
    h, w,c = 200, 180,3  #图片高度h，图片宽度w,图片rgb通道c
    y_true = []  # 真实值,这里真实值是指图片被分为10类，从0-9，每一个人一类，即一个人有20张表情，但是都为一类。
    y_pred = []  # 预测值，通过kmeans聚类，k=10,聚成10类人脸，每类人脸20个表情。
    acc_list = []
    nmi_list = []
    ari_list = []
    photos_data, y_true = get_face_images(PATH, ROW, COL) #photos_data为ndarray(200,108000)

    #主成分从1到n_components
    for i in range(1,n_components) :
        # reduced_data，降维后的数据，ndarray(200,i)，i个特征脸数据
        reduced_data = PCA(n_components=i,svd_solver='randomized').fit_transform(photos_data)
        y_pred = KMeans(n_clusters=10).fit_predict(reduced_data)
        acc, nmi, ari = clusteringMetrics(y_true, y_pred)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        print("i为{}时，ACC = {}, NMI = {}, ARI = {}".format(i,acc, nmi, ari))

    #显示pca降维后特征个数与聚类性能的关系柱状图
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    index = np.arange(9)
    rects1 = ax.bar(index, acc_list, bar_width,
                    alpha=opacity, color='b', error_kw=error_config,
                    label='ACC')

    rects2 = ax.bar(index + bar_width, nmi_list, bar_width,
                    alpha=opacity, color='r', error_kw=error_config,
                    label='NMI')
    rects2 = ax.bar(index + 2 * bar_width, ari_list, bar_width,
                    alpha=opacity, color='g', error_kw=error_config,
                    label='ARI')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9'))
    ax.legend()
    plt.show()





