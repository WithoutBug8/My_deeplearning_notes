import numpy as np
from common.functions import softmax, sigmoid, identity, Relu, cross_entropy_error
from common.gradient import numerial_gradient
from common.layers import *  # 引入神经网络的层结构
from collections import OrderedDict  # 引入有序字典，保存层级结构


class TwoLayerNet:
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 1. 将这些数据包装到字典里面
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 定义层结构
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 单独定义最后一层：SoftmaxWithLoss
        self.lastLayer = SoftmaxWithLoss()

    # 前向传播函数
    def forward(self, X):
        # 对于神经网络中每一层，直接调用layers中forward方法就可以了
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    # 损失函数
    def loss(self, x, t):
        y = self.forward(x)
        loss_value = self.lastLayer.forward(y, t)
        return loss_value

    # 计算准确度
    def accuracy(self, x, t):
        # 预测概率
        y_pred = self.forward(x)
        # 根据最大概率得到预测的分类号
        y = np.argmax(y_pred, axis=1)
        # 分类号与正确解标签对比，得到准确率
        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy

    # 计算梯度,使用数值微分方法
    def numerical_gradient(self, X, t):
        # 定义目标函数
        loss_f = lambda x: self.loss(X, t)
        # 对每个参数,使用数值微分方法计算梯度
        grads = {}
        grads['W1'] = numerial_gradient(loss_f, self.params['W1'])
        grads['b1'] = numerial_gradient(loss_f, self.params['b1'])
        grads['W2'] = numerial_gradient(loss_f, self.params['W2'])
        grads['b2'] = numerial_gradient(loss_f, self.params['b2'])
        return grads

    # 计算梯度，使用反向传播算法
    def gradient(self, X, t):
        # 前向传播,直到计算损失
        self.loss(X, t)
        # 反向传播
        dy = 1
        # 第一层，不计入for循环
        dy = self.lastLayer.backward(dy)
        # 逐层遍历，对神经网络中的所有层进行反向处理
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dy = layer.backward(dy)
        # 提取各个层的梯度
        grads = {}
        grads['W1'], grads["b1"] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads["b2"] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads
