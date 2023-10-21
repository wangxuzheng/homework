"""
3.给定的图像数据集，探讨LDA的降维效果。
"""
from utils import get_17flower_images
import matplotlib.pyplot as plt
from clustering_performance import cluster_acc
from utils import get_pasted_image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def LDA3():
    path = '17flowers/'
    ROW = 10
    COL = 10
    h = 200
    w = 200
    c = 3
    n_type = 17  #共有17种花
    IMAGE_SIZE = (h,w,c)
    flowers_data, y_true=  get_17flower_images(path,ROW,COL,h,w)
    background = get_pasted_image(flowers_data, IMAGE_SIZE, ROW, COL)
    plt.imshow(background)
    plt.show()


    #柱状图
    flowers_data, y_true = get_17flower_images(path,n_type,80,h,w)#获取完整图片集17类花，每种80张
    A_PCA = []  # acc of PCA
    A_LDA = []  # acc of LDA
    X_train, X_test, y_train, y_test = train_test_split(flowers_data, y_true, test_size=0.2, random_state=22)

    for i in range(1, n_type):  # LDA的n_components 不能大于min(n_features, n_classes - 1),
        # 这里n_class只有0-16,17种花。所以max_components = 16
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
    index = np.arange(n_type-1)
    ax.set_xticks(index + bar_width / 2)

    PCA_BAR = ax.bar(index, A_PCA, bar_width, alpha=opacity, color='b', label='PCA')
    LDA_BAR = ax.bar(index + bar_width, A_LDA, bar_width, alpha=opacity, color='g', label='LDA')

    label = []  # 横坐标标签
    for j in range(1, n_type):
        label.append(j)
    ax.set_xticklabels(label)
    plt.xlabel('Component')
    plt.ylabel('ACC')
    ax.legend()  # 图例标签
    plt.show()