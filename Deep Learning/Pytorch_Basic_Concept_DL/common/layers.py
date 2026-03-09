import numpy as np

from common.functions import *


# 定义ReLU
class ReLU:
    # 初始化
    def __init__(self):
        # 内部属性，记录那些x<=0
        self.mask = None

    # 前向传播过程
    def forward(self, x):
        self.mask = (x <= 0)
        # 大于零的不变
        y = x.copy()
        # 小于等于0的赋值为0
        y[self.mask] = 0
        return y

    # 反向传播过程
    def backward(self, dy):
        dx = dy.copy()
        # 将 x<=0 的值都赋值为0
        dx[self.mask] = 0
        return dx


# 定义sigmoid
class Sigmoid:
    # 初始化
    def __init__(self):
        # 定义内部属性，计算输出值y,用于反向传播时计算梯度
        self.y = None

    # 前向传播算法
    def forward(self, x):
        y = sigmoid(x)
        self.y = y
        return y

    # 定义反向传播算法
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        return dx


# 仿射层Affine
class Affine:
    # 初始化
    def __init__(self, W, b):
        self.W = W
        self.b = b
        # 对输入数据X进行保存，方便后续反向传播计算梯度
        self.X = None
        self.original_x_shape = None  # 原来的x输入的维度
        # 将权重和偏执参数的梯度(偏导数)保存成属性，方便梯度下降法计算
        self.dW = None
        self.db = None

    # 前向传播
    def forward(self, X):
        self.original_x_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        # self.X = X
        y = np.dot(self.X, self.W) + self.b
        return y

    # 反向传播
    def backward(self, dy):
        dx = np.dot(dy, self.W.T)
        dx = dx.reshape(*self.original_x_shape)
        self.dW = np.dot(self.X.T, dy)
        self.db = np.sum(dy, axis=0)
        return dx


# 输出层
class SoftmaxWithLoss:
    # 初始化方法
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    # 前向传播
    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    # 反向传播，默认初始值是1
    def backward(self, dy=1):
        n = self.t.shape[0]
        # 如果是one-hot编码的标签，直接带入公式计算
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 如果是顺序编码标签，需要单独进行计算;找到分类号对应的值，然后相减
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1

        return dx / n
