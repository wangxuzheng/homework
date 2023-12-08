import time
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from models import resnet18 as resnet
import matplotlib.pyplot as plt
import numpy as np


# 定义全局变量
path = '../DeepLearning/dataset'
n_epochs = 5  # epoch 的数目
batch_size = 20  # 决定每次读取多少图片
learning_rate = 0.02 #学习率
train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
print("GPU是否可用：", torch.cuda.is_available())  # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())  # 查看GPU数量
print("GPU索引号：", torch.cuda.current_device())  # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(0))  # 根据索引号得到GPU名称
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Current device:", device)

# 定义训练集个测试集
train_data = datasets.CIFAR10(root=path, train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.CIFAR10(root=path, train=False, transform=transforms.ToTensor(), download=False)

# 创建加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
# num_workers可以＞0,是调用系统子进程来跑,但是Windows可能会因为各种权限问题报错

# Model and optimizer
model = resnet()

#load the trained model
#Remember that you must call model.eval() to set dropout and batch normalization
#layers to evaluation mode before running inference.
#Failing to do this will yield inconsistent inference results.
# model = torch.load("model.pt")
# model.eval()

#load the model to gpu or cpu
model = model.to(device)

# SGD随机梯度下降法,lr学习率(步长),这里随机梯度和小批量随机梯度共用.SGD
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# 调用所有GPU
model = torch.nn.DataParallel(model)


# 训练神经网络
def train(epoch):
    correct = 0
    total = 0
    # 定义损失函数
    lossfunc = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 开始训练
    train_loss = 0.0
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        output = model(data)  # 得到预测值
        _, predicted = torch.max(output.data, 1)  # 前面一个返回值数据不是我们想要的
        correct += (predicted == labels).sum().item()
        loss = lossfunc(output, labels)  # 计算两者的误差
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        train_loss += loss.item() * labels.size(0)
    train_acc_list.append(correct / total)
    train_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch, train_loss))


# 在数据集上测试神经网络
def test():
    # 定义损失函数
    lossfunc = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # 测试集中不需要反向传播
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            total += labels.size(0)
            _, predicted = torch.max(output.data, 1)  # 前面一个返回值数据不是我们想要的
            correct += (predicted == labels).sum().item()
            #计算误差
            loss = lossfunc(output, labels)  # 计算两者的误差
            test_loss += loss.item() * labels.size(0)
    test_acc_list.append(correct / total)
    test_loss = test_loss /len(test_loader.dataset)
    test_loss_list.append(test_loss)
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


# 每训练一epoch,测试一下性能
t_total = time.time()
for epoch in range(n_epochs):
    train(epoch)
    current_accuracy = test()

# 保存模型
#torch.save(model, 'model.pt')
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# 画图
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="train accuracy")
plt.plot(np.arange(len(test_acc_list)), test_acc_list, label="test accuracy")
plt.legend()  # 显示图例
plt.xlabel('epoches')
plt.title('Model accuracy - train vs test')

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train loss")
plt.plot(np.arange(len(test_loss_list)), test_loss_list, label="test loss")
plt.legend()  # 显示图例
plt.xlabel('epoches')
plt.title('Model Loss - train vs test')


plt.show()
