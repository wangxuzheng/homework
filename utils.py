#工具
import os

import numpy as np
import pandas as pd
import matplotlib.image as imgplt


def get_face_images (path,n_people=10,n_faces=20):
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
    file = file[:n_people] #取前N个人
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
        img = imgplt.imread(file_uri)  # 图片shape为(200,180,3)
        img = img.reshape(1, -1)  # 将（200x180x3）的图片转为一维 108000
        img = pd.DataFrame(img)
        photos_data = pd.concat([photos_data, img], ignore_index=True)
    photos_data = photos_data.values  # shape为 (200, 200x180x3) 即 200个样本，108000 个特征。 values将dataframe转为ndarray
    return photos_data,y_true