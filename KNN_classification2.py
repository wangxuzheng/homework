"""
Sklearn中的datasets方法导入训练样本
并用留一法产生测试样本
用KNN分类并输出分类精度
"""
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt


def KNN2():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    loo = LeaveOneOut()  # 留一法，将数据集划分为训练集和测试集

    K = []
    Accuracy = []
    for k in range(1, 16): # 将k从1到16结束。
        correct = 0
        knn = KNeighborsClassifier(k)
        for train, test in loo.split(X):  # 对测试机和训练集进行分割
            knn.fit(X[train], y[train])  # 初始化knn进行训练。
            y_sample = knn.predict(X[test])
            if y_sample == y[test]:  # 如果是正确的就累积+1
                correct += 1
        K.append(k)
        Accuracy.append(correct / len(X))
        plt.plot(K, Accuracy)
        plt.xlabel('Accuracy:')
        plt.ylabel('K:')
        print('K次数:{} Accuracy正确率:{}'.format(k, correct / len(X)))

    plt.show()

