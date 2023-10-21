"""
2. 给定的图像数据集，探讨LDA的降维效果。
图片中彩色数字为训练集，黑白图片数字为测试集
"""


import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils import show_digit_image

def LDA2 ():
    scaler = MinMaxScaler()  # 归一化
    digits = load_digits()
    components = 2  #降为2维数据，刚好可以作为x,y在图表上显示
    X = digits.data   #(1797,64)
    y = digits.target
    images = digits.images  #(1797,8,8)
    X_pca = PCA(n_components=components).fit_transform(X)
    X_lda = LinearDiscriminantAnalysis(n_components=components).fit_transform(X, y)

    # 对每一个维度进行0-1归一化，不然点会很分散。彩色数字为训练集，黑白图片数字为测试集
    X_pca = scaler.fit_transform(X_pca)   #pca降维后的训练集 （1797，2）
    X_lda = scaler.fit_transform(X_lda)   #lda降维后的训练集 （1797，2）

    # 画图
    figure = plt.figure(figsize=(12, 6))
    ax1 = figure.add_subplot(1, 2, 1)
    ax2 = figure.add_subplot(1, 2, 2)

    #彩色数字
    for i in range(X.shape[0]):  #y_train.shape[0] == 1797
        # fontdict={'weight': 'bold', 'size': 9} 调整字体粗细
        ax1.text(X_pca[i, 0], X_pca[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),fontdict={'weight': 'bold', 'size': 9})
        ax2.text(X_lda[i, 0], X_lda[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),fontdict={'weight': 'bold', 'size': 9})

    # 黑白图片数字
    imageboxs_pca = show_digit_image(X_pca,images)#imagebox列表
    imageboxs_lda = show_digit_image(X_lda,images)#imagebox列表
    for imagebox in imageboxs_pca:
        ax1.add_artist(imagebox)
    for imagebox in imageboxs_lda:
        ax2.add_artist(imagebox)

    ax1.set_title("PCA")
    ax2.set_title("LDA")
    plt.show()

