import numpy as np
from common.functions import softmax, sigmoid, identity, Relu, cross_entropy_error
from common.gradient import numerial_gradient


class TwoLayerNet:
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 1. 将这些数据包装到字典里面
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 前向传播函数
    def forward(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        # 中间层
        a1 = X @ W1 + b1
        z1 = sigmoid(a1)
        a2 = z1 @ W2 + b2
        y = softmax(a2)
        return y

    # 损失函数
    def loss(self, x, t):
        y = self.forward(x)
        loss_value = cross_entropy_error(y, t)
        return loss_value

    # 计算准确度
    def accuracy(self, x, t):
        # 预测概率
        y_proba = self.forward(x)
        # 根据最大概率得到预测的分类号
        y = np.argmax(y_proba, axis=1)
        # 分类号与正确解标签对比，得到准确率
        accuracy = np.sum(y == t) / len(t)
        return accuracy

    # 计算梯度
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

