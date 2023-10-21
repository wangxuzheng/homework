"""
上机器学习数据库http://archive.ics.uci.edu/ml/index.php
下载并显示至少一个图像和一个文档数据集

"""

import os
import PIL.Image as Image
import pandas as pd
import matplotlib.pyplot as plt

def draw_image_data():
    COL = 10  # 指定拼接图片的列数
    ROW = 10  # 指定拼接图片的行数
    IMAGE_SIZE = (32,30) #图片尺寸32x30像素
    PATH = 'faces/'
    file_list = list(os.walk(PATH))
    person_name = file_list[0][1]
    file_list.pop(0)  # 去掉根目录内容
    file_uri_list = []
    for uri, dir, file_name_list in file_list:
        file_uri = []
        for file_name in file_name_list:
            file_uri.append(uri + '/' + file_name)  # 得到图片完整路径
        file_uri_list.append(file_uri)
    image_dict = dict(zip(person_name, file_uri_list))  # 名字：图像列表
    background = Image.new('RGB', (IMAGE_SIZE[0] * COL, IMAGE_SIZE[1] * ROW))
    for row in range(ROW):
        name = person_name.pop(0)
        for col in range(COL):
            background.paste(Image.open(image_dict[name][col]), (0 + IMAGE_SIZE[0] * col, 0 + IMAGE_SIZE[1] * row))
    plt.imshow(background)
    plt.show()




def draw_document_data():
    # 转化为dataframe格式
    iris_data = pd.read_table('iris/iris.txt', header=None, sep=',')
    # 为不同种类iris设置颜色进行区分
    Colors = []
    for i in range(iris_data.shape[0]):
        m = iris_data.iloc[i, -1]
        if m == 'Iris-setosa':
            Colors.append('green')
        elif m == 'Iris-versicolor':
            Colors.append('purple')
        elif m == 'Iris-virginica':
            Colors.append('red')
    # 绘图
    plt.subplot(1, 2, 1)
    plt.scatter(iris_data.iloc[:, 0], iris_data.iloc[:, 1], c=Colors, edgecolors='black')
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")

    plt.subplot(1, 2, 2)
    plt.scatter(iris_data.iloc[:, 0], iris_data.iloc[:, 2], c=Colors, edgecolors='black')
    plt.xlabel("sepal length")
    plt.ylabel("petal length")

    plt.show()
