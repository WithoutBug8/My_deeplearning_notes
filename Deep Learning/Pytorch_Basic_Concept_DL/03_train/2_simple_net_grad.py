import numpy as np

from common.functions import softmax, cross_entropy_error
from common.gradient import numerial_gradient


# 定义一个简单的神经网络类
class SimpleNet:
    # 初始化
    def __init__(self):
        self.W = np.random.randn(2, 3)

    # 前向传播
    def forward(self, X):
        a = X @ self.W
        y = softmax(a)
        return y

    # 计算损失值
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss


# 主程序类
if __name__ == '__main__':
    # 1. 定义数据
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    # 2. 定义神经网络模型
    net = SimpleNet()
    # 3. 计算梯度
    f = lambda w: net.loss(x, t)
    gradw = numerial_gradient(f, net.W)

    print(gradw)
