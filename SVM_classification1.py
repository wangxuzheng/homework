"""
各个分类器的数据可视化
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from clustering_performance import cluster_acc
import numpy as np


def classifier_comparison ():
    point_coordinate, point_type = make_circles(n_samples=800, noise=0.1, factor=0.1)
    X_train,X_test,y_train,y_test = train_test_split(point_coordinate, point_type, test_size=0.5, random_state=22)

    plt.figure(figsize=(12, 6))
    # 设置颜色
    z = np.array(y_test)
    colors = np.array(["green","red"])

    # 原始图
    plt.subplot(2, 3, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, marker="o", edgecolors='black', c=colors[z])
    plt.title('Original')


    # KNN
    plt.subplot(2, 3, 2)
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)
    y_sample = knn.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, marker="o", edgecolors='black', c=colors[y_sample])
    knn_acc = cluster_acc(y_test,y_sample)
    plt.title('KNN(acc={})'.format(knn_acc))


    # NaiveBayes
    plt.subplot(2, 3, 3)
    nb = naive_bayes.GaussianNB()
    nb.fit(X_train, y_train)
    y_sample = nb.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, marker="o", edgecolors='black', c=colors[y_sample])
    nb_acc = cluster_acc(y_test, y_sample)
    plt.title('NaiveBayes(acc={})'.format(nb_acc))

    # LogisticRegression
    plt.subplot(2, 3, 4)
    lr = LogisticRegression(max_iter=50000)
    lr.fit(X_train, y_train)
    y_sample = lr.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, marker="o", edgecolors='black', c=colors[y_sample])
    lr_acc = cluster_acc(y_test, y_sample)
    plt.title('LogisticRegression(acc={})'.format(lr_acc))

    # SVM
    plt.subplot(2, 3, 5)
    # (kernel) It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
    support= svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')
    support.fit(X_train, y_train)
    y_sample = support.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, marker="o", edgecolors='black', c=colors[y_sample])
    support_acc = cluster_acc(y_test, y_sample)
    plt.title('SVM(acc={})'.format(support_acc))

    #LinearRegreesion
    plt.subplot(2, 3, 6)
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_sample = linear.predict(X_test)  #线性回归返回的不是整数
    y_sample = np.round(y_sample)
    y_sample = y_sample.astype(int)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, marker="o", edgecolors='black', c=colors[y_sample])
    linear_acc = cluster_acc(y_test, y_sample)
    plt.title('LinearRegression(acc={})'.format(linear_acc))

    plt.show()