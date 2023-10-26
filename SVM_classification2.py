"""
给定的图像数据集，比较分类性能.
"""

from tabulate import tabulate
from utils import get_KNN_acc
from utils import get_GaussianNB_acc
from utils import get_local_sample
from utils import get_LinearRe_acc
from utils import get_LogisticRe_acc
from utils import get_SVM_acc

def LRandNBandKNNandSVM() :
    flowers, digits, face_images = get_local_sample()
    svm_kernel ='linear'
    # 17flowers分类
    flowers_KNN_acc = round(get_KNN_acc(flowers[0], flowers[1], flowers[2], flowers[3]), 4)
    flowers_GaussianNB_acc = round(get_GaussianNB_acc(flowers[0], flowers[1], flowers[2], flowers[3]), 4)
    flowers_LinearRe_acc = round(get_LinearRe_acc(flowers[0], flowers[1], flowers[2], flowers[3]), 4)
    flowers_LogisticRe_acc = round(get_LogisticRe_acc(flowers[0], flowers[1], flowers[2], flowers[3]), 4)
    flowers_SVM_acc = round(get_SVM_acc(svm_kernel,flowers[0], flowers[1], flowers[2], flowers[3]), 4)

    # digits分类
    digits_KNN_acc = round(get_KNN_acc(digits[0], digits[1], digits[2], digits[3]), 4)
    digits_GaussianNB_acc = round(get_GaussianNB_acc(digits[0], digits[1], digits[2], digits[3]), 4)
    digits_LinearRe_acc = round(get_LinearRe_acc(digits[0], digits[1], digits[2], digits[3]), 4)
    digits_LogisticRe_acc = round(get_LogisticRe_acc(digits[0], digits[1], digits[2], digits[3]), 4)
    digits_SVM_acc = round(get_SVM_acc(svm_kernel,digits[0], digits[1], digits[2], digits[3]), 4)

    # face images分类
    faces_KNN_acc = round(get_KNN_acc(face_images[0], face_images[1], face_images[2], face_images[3]), 4)
    faces_GaussianNB_acc = round(get_GaussianNB_acc(face_images[0], face_images[1], face_images[2], face_images[3]), 4)
    faces_LinearRe_acc = round(get_LinearRe_acc(face_images[0], face_images[1], face_images[2], face_images[3]), 4)
    faces_LogisticRe_acc = round(get_LogisticRe_acc(face_images[0], face_images[1], face_images[2], face_images[3]), 4)
    faces_SVM_acc = round(get_SVM_acc(svm_kernel,face_images[0], face_images[1], face_images[2], face_images[3]), 4)

    table = [['分类方法 \ 数据集', '17flowers', 'Digits', 'Face images'],
             ['KNN', str(flowers_KNN_acc), str(digits_KNN_acc), str(faces_KNN_acc)],
             ['NaiveBayes',str(flowers_GaussianNB_acc), str(digits_GaussianNB_acc),str(faces_GaussianNB_acc)],
             ['LinearRegression', str(flowers_LinearRe_acc), str(digits_LinearRe_acc), str(faces_LinearRe_acc)],
             ['LogisticRegression',str(flowers_LogisticRe_acc),str(digits_LogisticRe_acc), str(faces_LogisticRe_acc)],
             ['SVM',str(flowers_SVM_acc),str(digits_SVM_acc), str(faces_SVM_acc)]
             ]
    table = tabulate(table)
    print(table)

