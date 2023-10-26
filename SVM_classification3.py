"""
给定的图像数据集CIFAI-10，比较分类性能.
"""

import numpy as np
from utils import unpickle, get_SVM_acc
from utils import get_KNN_acc
from utils import get_GaussianNB_acc
from utils import get_LinearRe_acc
from utils import get_LogisticRe_acc
from tabulate import tabulate
import time

def comparision_CIFAR10 () :
    start_time = time.time()
    path = 'dataset/'
    svm_kernel = 'poly'
    part = 10000  #选择训练的图片数量 从0-10000
    test_data = unpickle(path + 'test_batch')
    X_test, y_test = test_data[b'data'][0:part], np.array(test_data[b'labels'][0:part])
    ACC_all = []
    for i in range(1, 4):#从batch 1 - 3
        train_data = unpickle(path + 'data_batch_' + str(i))
        X_train, y_train = train_data[b'data'][0:part], np.array(train_data[b'labels'][0:part])
        KNN_acc = round(get_KNN_acc(X_train, X_test, y_train, y_test), 4)
        GaussianNB_acc = round(get_GaussianNB_acc(X_train, X_test, y_train, y_test),4)
        LinearRe_acc = round(get_LinearRe_acc(X_train, X_test, y_train, y_test), 4)
        LogisticRe_acc = round(get_LogisticRe_acc(X_train, X_test, y_train, y_test),4)
        SVM_acc = round(get_SVM_acc(svm_kernel,X_train, X_test, y_train, y_test),4)
        batch_result = [str(KNN_acc),str(GaussianNB_acc),str(LinearRe_acc),str(LogisticRe_acc),str(SVM_acc)]
        ACC_all.append(batch_result)
    table = [['分类方法\数据集', 'batch_1', 'batch_2', 'batch_3'],
             ['KNN',ACC_all[0][0], ACC_all[1][0], ACC_all[2][0]],
             ['NaiveBayes',ACC_all[0][1], ACC_all[1][1],ACC_all[2][1]],
             ['LinearRegression', ACC_all[0][2], ACC_all[1][2],ACC_all[2][2]],
             ['LogisticRegression',ACC_all[0][3],ACC_all[1][3], ACC_all[2][3]],
             ['SVM',ACC_all[0][4],ACC_all[1][4], ACC_all[2][4]]
             ]
    table = tabulate(table)
    print(table)
    end_time = time.time()
    print('程序运行时间为: %s Seconds' % (end_time - start_time))