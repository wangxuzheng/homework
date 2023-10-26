"""
1. 给定的图像数据集，可视化并输出聚类性能。(和clustering5.py一样，这里不展示)

2. 给定的图像数据集，计算相应的特征脸(eigenfaces),(RGB)（特征脸是指的所有脸的特征脸，不是指某一个人的特征脸）


"""

import matplotlib.pyplot as plt
from utils import get_face_images
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def eigenfaces_RGB():
    scaler = MinMaxScaler()
    ROW = 10  # 指定拼接图片的行数,即有多少人
    COL = 20  # 指定拼接图片的列数，即每个人有多少图片
    PATH = 'face_images/'
    n_components = 10  #用PCA降维，n_components为降到的维数   主成分数量
    h, w,c = 200, 180,3  #图片高度h，图片宽度w,图片rgb通道c
    y_true = []  # 真实值,这里真实值是指图片被分为10类，从0-9，每一个人一类，即一个人有20张表情，但是都为一类。
    y_pred = []  # 预测值，通过kmeans聚类，k=10,聚成10类人脸，每类人脸20个表情。
    photos_data, y_true = get_face_images(PATH, ROW, COL) #photos_data为ndarray(200,108000)


    pca = PCA(n_components=n_components,svd_solver='randomized').fit(photos_data)
    components = pca.components_#pca的10个主成分（10，108000），即10个主要的新特征（新基向量）
    components = scaler.fit_transform(components)  # MinMaxScaler expect array with dim <= 2.
    eigenfaces = components.reshape(n_components,h,w,c)
    # 画特征脸
    fig, ax = plt.subplots(nrows=1,ncols=n_components, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    count = 0
    for i in range(n_components):
        eigenface = eigenfaces[count]
        ax[i].imshow(eigenface)
        count += 1
    plt.xticks([])  # 去除X轴坐标
    plt.yticks([])  # 去除X轴坐标
    # acc, nmi, ari = clusteringMetrics(y_true, y_pred)
    # plt.suptitle('ACC = {}, NMI = {}, ARI = {}'.format(acc, nmi, ari))
    plt.show()



