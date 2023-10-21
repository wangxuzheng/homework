"""

给定的图像数据集，可视化并输出聚类性能。

"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from clustering_performance import clusteringMetrics
from utils import get_face_images
from utils import get_pasted_image

def Kmeans5():
    ROW = 10  # 指定拼接图片的行数,即有多少人
    COL = 20  # 指定拼接图片的列数，即每个人有多少图片
    PATH = 'face_images/'
    y_true = []   #真实值,这里真实值是指图片被分为10类，从0-9，每一个人一类，即一个人有20张表情，但是都为一类。
    y_pred = []   #预测值，通过kmeans聚类，k=10,聚成10类人脸，每类人脸20个表情。
    n_samples, h, w, c = ROW * COL, 200, 180, 3  # n个样本，图片高度h，图片宽度w,图片rgb通道c
    IMAGE_SIZE = (h, w, c)  #图片尺寸
    photos_data,y_true = get_face_images(PATH,ROW,COL)
    cluster = KMeans(n_clusters=ROW) .fit(photos_data) # 聚类成10个人（样本为10个人，每个人20张表情图片）
    y_pred = cluster.labels_
    background = get_pasted_image(photos_data,IMAGE_SIZE,ROW,COL)
    plt.imshow(background)
    acc, nmi, ari = clusteringMetrics(y_true, y_pred)
    plt.suptitle('ACC = {}, NMI = {}, ARI = {}'.format(acc, nmi, ari))
    plt.show()

