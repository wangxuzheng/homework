"""
1. 给定的图像数据集，探讨LDA的降维效果。

"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from clustering_performance import cluster_acc
import numpy as np
from utils import get_pasted_image

def LDA1():
    digits = load_digits() # 总样本数1797  特征数64=8×8
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=22)    # 训练样本数1797  特征数64=8×8
    IMAGE_DATA = digits.images
    ROW = 15
    COL = 25
    IMAGE_SIZE = (8,8) #图片像素8x8
    background = get_pasted_image(IMAGE_DATA, IMAGE_SIZE, ROW, COL)
    plt.imshow(background)


    A_PCA = []  # acc of PCA
    A_LDA = []  # acc of LDA
    for i in range(1, 10):  # LDA的n_components 不能大于min(n_features, n_classes - 1),
        # 这里n_class只有0-9，十个类别。所以max_components = 9
        # LDA降维最多降到类别数k-1的维数。由于投影矩阵W是一个利用了样本的类别得到的投影矩阵（n*d,一般d<<n）
        # PCA + KNN
        pca = PCA(n_components=i).fit(X_train)  # pca模型训练
        # 将输入数据投影到特征面正交基上
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        knn = KNeighborsClassifier()
        knn.fit(X_train_pca, y_train)
        y_pred = knn.predict(X_test_pca)
        acc_pca = cluster_acc(y_test, y_pred)
        A_PCA.append(acc_pca)
        # LDA + KNN
        lda = LinearDiscriminantAnalysis(n_components=i).fit(X_train, y_train)  # lda模型训练 记得加上y_train训练集的标签
        # 将输入数据投影到特征面正交基上
        X_train_lda = lda.transform(X_train)
        X_test_lda = lda.transform(X_test)
        knn = KNeighborsClassifier()
        knn.fit(X_train_lda, y_train)
        y_pred = knn.predict(X_test_lda)
        acc_lda = cluster_acc(y_test, y_pred)
        A_LDA.append(acc_lda)

    # 画柱状图
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.6  # 不透明度
    index = np.arange(9)
    ax.set_xticks(index + bar_width / 2)

    PCA_BAR = ax.bar(index, A_PCA, bar_width, alpha=opacity, color='b', label='PCA')
    LDA_BAR = ax.bar(index + bar_width, A_LDA, bar_width, alpha=opacity, color='g', label='LDA')

    label = []  # 横坐标标签
    for j in range(1, 10):
        label.append(j)
    ax.set_xticklabels(label)
    plt.xlabel('Component')
    plt.ylabel('ACC')
    ax.legend()  # 图例标签
    plt.show()
