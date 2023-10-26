"""
给定任意邮件（一段文档），输出是否为垃圾邮件.
"""
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from clustering_performance import cluster_acc
from SpamEmailDetector import EmailFeatureGeneration as Email


def spam_detection_comparison():
    X, Y = Email.Text2Vector()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=22)
    # print("X_train.shape =", X_train.shape)
    # print("X_test.shape =", X_test.shape)

    # 朴素贝叶斯
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_sample_bayes = nb.predict(X_test)
    Bayes_ACC = cluster_acc(y_test, y_sample_bayes)
    print("Bayes_ACC =", Bayes_ACC)

    fig = plt.figure()
    plt.subplot(121)
    plt.title('Bayes')
    confusion = confusion_matrix(y_sample_bayes, y_test)
    confusion = confusion / X_test.shape[0]
    # print(confusion)
    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.3g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_sample_knn = knn.predict(X_test)
    KNN_ACC = cluster_acc(y_test, y_sample_knn)
    print("KNN_ACC =", KNN_ACC)

    plt.subplot(122)
    plt.title('KNN')
    confusion = confusion_matrix(y_sample_knn, y_test)
    confusion = confusion / X_test.shape[0]
    sns.heatmap(confusion, annot=True, cmap='YlGn', fmt='.3g')
    plt.xlabel('Predicted label')

    plt.show()
