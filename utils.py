# 工具
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import offsetbox
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from sklearn import naive_bayes
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from clustering_performance import cluster_acc
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer


def get_face_images(path, n_people=10, n_faces=20):
    """
    get face images data

    # Arguments
       n_people: number of people,即有多少个人,默认10个人
       n_faces: number of faces,即每个人有多少张图片，默认20张图片
        path  :path of images
    # Return
        photos_data : ndarray, (样本数，特征数)即（n_people * n_faces, height * width * channel）
        y_true : list,  lize.size() = n_people * n_faces, 每个元素的range( 0 ,n_people)
    """
    n_people = n_people  # number of people,即有多少个人
    n_faces = n_faces  # number of faces,即每个人有多少张图片
    PATH = path
    file_uri_list = []
    y_true = []  # 真实值,这里真实值是指图片被分为10类，从0-9，每一个人一类，即一个人有20张表情，但是都为一类。
    photos_data = pd.DataFrame()
    # 获取图片
    file = os.listdir(PATH)
    file = file[:n_people]  # 取前N个人
    i = 0
    for subfile in file:
        photo_list = os.listdir(PATH + subfile)
        photo_list.sort(key=lambda x: int(x.split('.')[1]))  # 按名称序号排序
        photo_list = photo_list[:n_faces]  # 取前N张脸
        for photo_name in photo_list:
            file_uri_list.append(PATH + subfile + '/' + photo_name)
            y_true.append(i)  # 真实值,这里真实值是指图片被分为10类，从0-9，每一个人一类。
        i += 1
    for file_uri in file_uri_list:
        img = plt.imread(file_uri)  # 图片shape为(200,180,3)
        #img = rgb2gray(img)  #转为黑白图
        img = img.reshape(1, -1)  # 将（200x180x3）的图片转为一维 108000
        img = pd.DataFrame(img)
        photos_data = pd.concat([photos_data, img], ignore_index=True)
    photos_data = photos_data.values  # shape为 (200, 200x180x3) 即 200个样本，108000 个特征。 values将dataframe转为ndarray
    return photos_data, y_true


def show_digit_image(X_data, digit_images):
    """
    在plt上显示黑白数字小图片

    # Arguments
       X_data: (n_sample,2),n个样本，2个特征数据，即X和Y坐标。（降维后的数据）
       digit_images:(n_sample,8,8) ，datasets.load_digits().images数据 （n个样本,8,8） 即n个样本,8x8像素
    # Return
        imageboxs ： list of imagebox,AnnotationBbox,黑白数字小图片
    """
    imageboxs = []
    X = X_data  # 降维后的数据
    images = digit_images  # 数字图片
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # 假设最开始出现的缩略图在(1,1)位置上
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)  # 算出样本点与所有展示过的图片（shown_images）的距离
            if np.min(dist) < 4e-3:  # 若最小的距离小于4e-3，即存在有两个样本点靠的很近的情况，则通过continue跳过展示该数字图片缩略图
                continue
            shown_images = np.r_[shown_images, [X[i]]]  # 展示缩略图的样本点通过纵向拼接加入到shown_images矩阵中

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r), X[i])
            imageboxs.append(imagebox)
    return imageboxs

def get_17flower_images(path, n_type=17, n_flower=80,height=200,width=180):
    """
    get 17flower images data

    # Arguments
       n_type: number of type of flowers,即有多少类花,图片集总共有17类。
       n_flower: number of flowers,即每种花有多个张图片，默认80张。
       height :图片高度像素，默认为200
       width :图片宽度像素,默认为180
        path  :path of images
    # Return
        flowers_data : ndarray, (样本数，特征数)即（n_type * n_flower, height * width * channel）
        y_true : list,  lize.size() = n_type * n_flower, 每个元素的range( 0 ,n_people)
    """
    flowers_data = pd.DataFrame()
    y_true = []  # 真实值,这里真实值是指图片被分为n_type类，从0-n_type，
    file_uri_list = []  # 图片uri列表
    h = height
    w = width

    # 总数据集限制，共17类，每类80张图片，超过就报错。
    if (n_type > 17 or n_flower > 80):
        return -1

    # 获取图片
    file = os.listdir(path)
    i = 0  # 图片编码计数，从1到1360，每80为一类，即1-80为类别0，81-160为类别1
    for subfile in file:
        file_uri_list.append(path + subfile)
        y_true.append(i // 80)  # 向下取整，79/80 = 0.9875 取整为0,即0-79为第0类
        i = i + 1
    # 只要其中前n_type类和每类的前n_flower张图片,和对应的ytrue
    truncated_image_list = []  #截取后的图片uri地址列表
    truncated_y_true_list =[]  #截取后的y_true地址列表
    for ind in range(n_type):
        truncated_image_list.extend(file_uri_list[ind * 80:ind * 80 + n_flower])
        truncated_y_true_list.extend(y_true[ind * 80:ind * 80 + n_flower])
    for file_uri in truncated_image_list:
        original = Image.open(file_uri)
        img = original.resize((w, h), Image.NEAREST)  # 统一所有图片像素为(w,h)
        img = asarray(img)  # 转为ndarray即（200，180，3）
       # img = rgb2gray(img)  #转为黑白图
        img = img.reshape(1, -1)  # 将（200x180x3）的图片转为一维 108000
        img = pd.DataFrame(img)
        flowers_data = pd.concat([flowers_data, img], ignore_index=True)
    flowers_data = flowers_data.values  # shape为 (样本数,特征数) 即200x180x3个特征。 values将dataframe转为ndarray
    return flowers_data, truncated_y_true_list

#将彩色图像转为灰度图片  ，Gray = R * 0.299 + G * 0.587 + B * 0.114
def rgb2gray(color_image):
    gray_image = []
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            #使用转换公式进行彩色灰度转换
            gray_image.append(int(color_image[i,j,0]*0.3+color_image[i,j,1]*0.59+color_image[i,j,2]*0.11))
    return np.array(gray_image).reshape(color_image.shape[0],color_image.shape[1])

# 将多张图片拼接成一张图，返回拼接后的图
def get_pasted_image(IMAGE_DATA, IMAGE_SIZE, ROW, COL):
    """
    IMAGE_DATA 为 （样本数,特征数） 或(样本数,h,w,c) 或(样本数,h,w)
    """
    if (len(IMAGE_DATA.shape) ==2): #若输入的IMAGE_DATA为(样本数,特征数)，则要转为原始图片(样本数,图片像素)，否则直接使用
        #判断是彩色还是黑白图片
        if len(IMAGE_SIZE) == 3 : #彩色图片为（h,w,c）
            # 将输入的图片数据 (样本数,特征数)，转为(样本数,h*w*c)，
            IMAGE_DATA = IMAGE_DATA.reshape(COL * ROW, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])
            background = Image.new('RGB', (IMAGE_SIZE[1] * COL, IMAGE_SIZE[0] * ROW), "white")
        elif len(IMAGE_SIZE) == 2: #黑白图片为(h,w)
            # 将输入的图片数据 (样本数,特征数)，转为(样本数,h*w)，
            IMAGE_DATA = IMAGE_DATA.reshape(COL * ROW, IMAGE_SIZE[0], IMAGE_SIZE[1])
            background = Image.new('L', (IMAGE_SIZE[1] * COL, IMAGE_SIZE[0] * ROW), "white")
    else:
        if len(IMAGE_SIZE) == 3:  # 彩色图片为（h,w,c）
            background = Image.new('RGB', (IMAGE_SIZE[1] * COL, IMAGE_SIZE[0] * ROW), "white")
        elif len(IMAGE_SIZE) == 2:  # 黑白图片为(h,w)
            background = Image.new('L', (IMAGE_SIZE[1] * COL, IMAGE_SIZE[0] * ROW), "white")

            # size(width,height)
    i = 0
    for row in range(ROW):
        for col in range(COL):
            background.paste(Image.fromarray(IMAGE_DATA[i]), (IMAGE_SIZE[1] * col, IMAGE_SIZE[0] * row))
            i = i + 1
    return background

#获取本地测试数据（17flowers、digits、和face images）用于后续分类方法比较

def get_local_sample ():
    # 读取数据
    path_flower = '17flowers/'
    path_face = 'face_images/'
    digits = load_digits()
    photos_data, photos_true = get_face_images(path_face, 10, 20)  # 只选取10个人，每个人20张图片
    flowers_data, flowers_true = get_17flower_images(path_flower, 10, 20)  # 只选取10类花，每种20张图片
    # flowers_data, flowers_true = get_17flower_images(path_flower, 17, 80, 200,200)  # 获取完整图片集17类花，每种80张
    digits_data = digits.data
    digits_true = digits.target

    # 拆分训练集和数据集
    X_train_flowers, X_test_flowers, y_train_flowers, y_test_flowers = \
        train_test_split(flowers_data, flowers_true, test_size=0.2, random_state=22)
    X_train_digits, X_test_digits, y_train_digits, y_test_digits = \
        train_test_split(digits_data, digits_true, test_size=0.2, random_state=22)
    X_train_faces, X_test_faces, y_train_faces, y_test_faces = \
        train_test_split(photos_data, photos_true, test_size=0.2, random_state=22)


    flowers = [X_train_flowers, X_test_flowers, y_train_flowers, y_test_flowers]
    digits = [X_train_digits, X_test_digits, y_train_digits, y_test_digits]
    face_images = [X_train_faces, X_test_faces, y_train_faces, y_test_faces]

    return flowers,digits,face_images


def unpickle(file):  # 官方给的例程
    with open(file, 'rb') as fo:
        cifar = pickle.load(fo, encoding='bytes')
    return cifar


# KNN分类器
def get_KNN_acc(*data):
    X_train, X_test, y_train, y_test = data
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_sample = knn.predict(X_test)
    ACC = cluster_acc(y_test, y_sample)
    return ACC


# 高斯贝叶斯分类器  ,高斯朴素贝叶斯可用于 特征为连续值 的分类问题，比如iris分类
"""
使用CategoricalNB时会遇到问题，
CategoricalNB expects a certain number of classes in the feature vectors during training and testing
The train_test_split function does not guarantee this,
"""
def get_GaussianNB_acc(*data):
    X_train, X_test, y_train, y_test = data
    nb = naive_bayes.GaussianNB()  # ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB','CategoricalNB']
    nb.fit(X_train, y_train)
    # ACC = nb.score(X_test, y_test)
    y_sample = nb.predict(X_test)
    ACC = cluster_acc(y_test, y_sample)
    return ACC

# 逻辑回归分类器
def get_LogisticRe_acc(*data):
    X_train, X_test, y_train, y_test = data
    lr = LogisticRegression(max_iter=50000)
    lr.fit(X_train, y_train)
    y_sample = lr.predict(X_test)
    ACC = cluster_acc(y_test, y_sample)
    return ACC

def get_LinearRe_acc(*data):
    X_train, X_test, y_train, y_test = data
    labelbin = LabelBinarizer() # 转为one_hot
    y_one_hot = labelbin.fit_transform(y_train) #(160,10) ,y_train为(160,1)
    linear = LinearRegression()
    linear.fit(X_train, y_one_hot)
    y_sample = linear.predict(X_test) #线性回归的分类，因为返回的是连续的数值（每个类别的概率，并非类别标签）,
    # 比如每个样本在每个类别（0-9）有概率（0.55，-0.12,...,0.32）十个，用argmax取概率最大的那个索引（索引即类别0-9）
    b =y_sample.argmax(axis=1)#返回数组沿着某一条轴最大值的索引
    y_sample = labelbin.inverse_transform(y_sample)
    ACC = cluster_acc(y_test, y_sample)
    return ACC


def get_SVM_acc(kernel,*data):
    X_train, X_test, y_train, y_test = data
    # (kernel) It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
    support= svm.SVC(C=2, kernel=kernel, gamma=10, decision_function_shape='ovo')
    support.fit(X_train, y_train)
    y_sample = support.predict(X_test)
    ACC = cluster_acc(y_test, y_sample)
    return ACC
