"""

给定的图像数据集，可视化并输出聚类性能。

"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from clustering_performance import clusteringMetrics
from utils import get_face_images

def Kmeans5():
    ROW = 10  # 指定拼接图片的行数,即有多少人
    COL = 20  # 指定拼接图片的列数，即每个人有多少图片
    PATH = 'face_images/'
    y_true = []   #真实值,这里真实值是指图片被分为10类，从0-9，每一个人一类，即一个人有20张表情，但是都为一类。
    y_pred = []   #预测值，通过kmeans聚类，k=10,聚成10类人脸，每类人脸20个表情。
    n_samples, h, w, c = ROW * COL, 200, 180, 3  # n个样本，图片高度h，图片宽度w,图片rgb通道c
    photos_data,y_true = get_face_images(PATH,ROW,COL)
    cluster = KMeans(n_clusters=ROW) .fit(photos_data) # 聚类成10个人（样本为10个人，每个人20张表情图片）
    y_pred = cluster.labels_
    centers = cluster.cluster_centers_#shape为 (10 , 200x180x3) 即 10 个中心点（即10人个），108000 个特征
    result = centers[y_pred]  #即将同一人的20张表情图片统一为一张代表性图片
    result = result.astype("int64")
    result = result.reshape(ROW*COL, h, w, c) #200张图片,200高,180宽,3通道
    #原始图片
    original = photos_data.reshape(ROW*COL, h, w, c)

    # 画图
    fig,ax  = plt.subplots(nrows=ROW,ncols=COL,sharex = True,sharey = True,figsize = [15,8],dpi = 80)
    plt.subplots_adjust(wspace = 0,hspace = 0)
    count = 0
    for i in range(ROW):
        for j in range(COL):
            ax[i,j].imshow(original[count])
            count += 1
    plt.xticks([])#去除X轴坐标
    plt.yticks([])#去除X轴坐标
    acc, nmi, ari = clusteringMetrics(y_true, y_pred)
    plt.suptitle('ACC = {}, NMI = {}, ARI = {}'.format(acc, nmi, ari))
    plt.show()

