import torch
import torch.nn.functional as F  # 激励函数的库


# 建立一个三层感知机网络
class MLP(torch.nn.Module):  # 继承 torch 的 Module
    data_dimension = 32*32*3   # 28*28 for mnist , 32*32*3 for cifar10
    def __init__(self):
        super(MLP, self).__init__()
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(MLP.data_dimension, 512)  # 第一个隐含层
        self.fc2 = torch.nn.Linear(512, 128)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(128, 10)  # 输出层

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(din.size(0), MLP.data_dimension)  # .view( )是一个tensor的方法,使得tensor改变size但是元素的总数是不变的
        # din = din.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))  # 隐层激活函数 relu
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)  # 输出层激活函数 softmax
        # 10个数字实际上是10个类别,输出是概率分布,最后选取概率最大的作为预测值输出
        return dout
