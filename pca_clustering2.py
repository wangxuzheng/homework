"""
 给定的图像数据集，计算相应的特征脸(eigenfaces),(GRAY)
"""

import matplotlib.pyplot as plt
from utils import get_face_images
from sklearn.decomposition import PCA
import numpy as np


#将彩色图像转为灰度图片  ，Gray = R * 0.299 + G * 0.587 + B * 0.114
def rgb2gray(color_image):
    gray_image = []
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            #使用转换公式进行彩色灰度转换
            gray_image.append(int(color_image[i,j,0]*0.3+color_image[i,j,1]*0.59+color_image[i,j,2]*0.11))
    return np.array(gray_image).reshape(200,180)


def eigenfaces_GRAY():
    ROW = 10  # 指定拼接图片的行数,即有多少人
    COL = 20  # 指定拼接图片的列数，即每个人有多少图片
    PATH = 'face_images/'
    n_components = 10  #用PCA降维，n_components为降到的维数   主成分数量
    h, w, = 200, 180  #图片高度h，图片宽度w
    y_true = []  # 真实值,这里真实值是指图片被分为10类，从0-9，每一个人一类，即一个人有20张表情，但是都为一类。
    y_pred = []  # 预测值，通过kmeans聚类，k=10,聚成10类人脸，每类人脸20个表情。
    photos_data, y_true = get_face_images(PATH, ROW, COL) #photos_data为ndarray(200,108000)，108000即200x180x3原始rgb图片
    photos_data = photos_data.reshape(200,200,180,3)#恢复原始图片集格式
    gray_images_data = []
    for photo_data in photos_data :
        gray_image = rgb2gray(photo_data)
        gray_image_data = gray_image.reshape(1, -1)  # 将（200x180）的图片转为一维 36000
        gray_images_data.append(gray_image_data)
    gray_images_data = np.array(gray_images_data).reshape(200,36000)#list变为ndarray（200，36000）
    gray_images_data = gray_images_data.T    #转置变为(36000,200)
    # reduced_data，降维后的数据，ndarray(36000,10)，10个特征脸数据
    reduced_data = PCA(n_components=n_components,svd_solver='randomized').fit_transform(gray_images_data)
    reduced_data = reduced_data.T #转置为(10,36000)
    eigenfaces = reduced_data.reshape(n_components,200,180) #重新变为10个特征脸的图片

    # 画特征脸
    fig, ax = plt.subplots(nrows=1,ncols=n_components, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    count = 0
    for i in range(n_components):
        eigenface = eigenfaces[count]
        ax[i].imshow(eigenface,cmap=plt.cm.gray)
        count += 1
    plt.xticks([])  # 去除X轴坐标
    plt.yticks([])  # 去除X轴坐标
    # acc, nmi, ari = clusteringMetrics(y_true, y_pred)
    # plt.suptitle('ACC = {}, NMI = {}, ARI = {}'.format(acc, nmi, ari))
    plt.show()



