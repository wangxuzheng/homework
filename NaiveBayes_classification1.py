"""
给定的图像数据集，比较分类性能.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from utils import get_face_images
from utils import get_17flower_images
from utils import get_KNN_acc
from utils import get_GaussianNB_acc


def NaiveBayesAndKNN():
    # 读取数据
    path_flower = '17flowers/'
    path_face = 'face_images/'
    digits = load_digits()
    photos_data, photos_true = get_face_images(path_face, 10, 20)  # 只选取10个人，每个人20张图片
    flowers_data, flowers_true = get_17flower_images(path_flower, 10, 20)  # 只选取10类花，每种20张图片
    digits_data = digits.data
    digits_true = digits.target

    # 拆分训练集和数据集
    X_train_digits, X_test_digits, y_train_digits, y_test_digits = \
        train_test_split(digits_data, digits_true, test_size=0.2, random_state=22)
    X_train_faces, X_test_faces, y_train_faces, y_test_faces = \
        train_test_split(photos_data, photos_true, test_size=0.2, random_state=22)
    X_train_flowers, X_test_flowers, y_train_flowers, y_test_flowers = \
        train_test_split(flowers_data, flowers_true, test_size=0.2, random_state=22)

    # 17flowers分类
    flowers_KNN_acc = round(get_KNN_acc(X_train_flowers, X_test_flowers, y_train_flowers, y_test_flowers),4)
    flowers_GaussianNB_acc = round(get_GaussianNB_acc(X_train_flowers, X_test_flowers, y_train_flowers, y_test_flowers),4)
    # digits分类
    digits_KNN_acc = round(get_KNN_acc(X_train_digits, X_test_digits, y_train_digits, y_test_digits),4)
    digits_GaussianNB_acc = round(get_GaussianNB_acc(X_train_digits, X_test_digits, y_train_digits, y_test_digits),4)
    # face images分类
    faces_KNN_acc = round(get_KNN_acc(X_train_faces, X_test_faces, y_train_faces, y_test_faces),4)
    faces_GaussianNB_acc = round(get_GaussianNB_acc(X_train_faces, X_test_faces, y_train_faces, y_test_faces),4)

    table = [['分类方法 \ 数据集 ', ' 17flowers', 'Digits ', 'Face images'],
             ['KNN            ', '   '+str(flowers_KNN_acc), ''+str(digits_KNN_acc),'  '+str(faces_KNN_acc)],
             ['NaiveBayes     ', '   '+str(flowers_GaussianNB_acc),''+ str(digits_GaussianNB_acc),'  '+str(faces_GaussianNB_acc)]]
    for row in table:
        for col in row:
            print(col, end='\t')
        print()
