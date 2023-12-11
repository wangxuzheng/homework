import torch.nn as nn
import torch.nn.functional as F  # 激励函数的库

#首先经过卷积层，进行特征提取，经过特征提取之后，图片信息变成了一个向量；而后将向量接入一个全连接网络进行分类处理。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)  #3通道，10通道，卷积核大小=5x5,(32-5+1) = 28
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(500, 10)


    def forward(self, din):
        # 前向传播， 输入值：din（20，3，32，32）, 返回值 dout
        batch_size = din.size(0)
        x = F.relu(self.conv1(din)) #X变成(20,10,28,28)
        x = self.pooling(x)  #池化后变成(20,10,14,14)
        x = F.relu(self.conv2(x)) #X变成(20,20,10,10)
        x = self.pooling(x)  #池化后变成(20,20,5,5)
        x = x.view(batch_size, -1)
        dout = self.fc(x)
        return dout


