"""
给定的图像，对其像素进行聚类并可视化
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def Kmeans4():
    # 读取原始图像
    path = "stones.jpg"
    original = plt.imread(path)
    # 除以255，将像素点的值变为0-1之间
    original = original / 255
    # height*width*channel = 308*468*3
    h, w, c = original.shape
    X = original.reshape(h * w, c)
    # 聚类数到6
    k = 6
    fig = plt.figure()
    f = fig.add_subplot(231)
    f.set_title("original")
    f.imshow(original)
    for i in range(2, k + 1):
        cluster = KMeans(n_clusters=i, random_state=9).fit(X)
        centroid = cluster.cluster_centers_  # 查看聚类后的质心
        y_pred = cluster.labels_  # 获取训练后对象的每个样本的标签
        # 重新变成图片308*468
        y_pred = y_pred.reshape(h, w)
        index = '3{}{}'.format(str(int(k / 2)), str(i))
        f = fig.add_subplot(int(index))
        f.set_title('K = {}'.format(i))
        f.imshow(y_pred)
    plt.show()


# 显示当k=n时压缩图片的效果
def image_compress(original, k=6):
    h, w, c = original.shape
    X = original.reshape(h * w, c)

    # k = n时的压缩图片
    cluster = KMeans(n_clusters=k, random_state=9).fit(X)
    centroid = cluster.cluster_centers_  # 查看聚类后的质心
    labels = cluster.labels_  # 获取训练后对象的每个样本的标签
    # 按照h,w,c创建一个空白图片
    img = np.zeros((h, w, c))
    label_index = 0
    # 通过for循环，遍历img中每一个点，并且从labels中取出下标对应的聚类重新给img赋值
    for i in range(h):
        for j in range(w):
            img[i][j] = centroid[labels[label_index]]
            label_index += 1
    return img


def show_compressed_image(path="stones.jpg", k=6):
    original = plt.imread(path)
    # 除以255，将像素点的值变为0-1之间
    original = original / 255
    # height*width*channel
    h, w, c = original.shape
    X = original.reshape(h * w, c)
    # 新建一个图片框
    fig = plt.figure()

    # 原始图片
    f = fig.add_subplot(121)
    f.set_title("original")
    f.imshow(original)

    # k = n时的压缩图片
    img = image_compress(original, k)
    f = fig.add_subplot(122)
    f.set_title("compressed image when k={}".format(k))
    f.imshow(img)
    plt.show()


def save_original_and_compressed_image(path="stones.jpg", k=6):
    original = plt.imread(path)
    original = original/255
    plt.title('original')
    plt.imshow(original)
    plt.savefig("1.png")

    compressed_image = image_compress(original,k)
    plt.title('compressed_image')
    plt.imshow(compressed_image)
    plt.savefig("2.png")
