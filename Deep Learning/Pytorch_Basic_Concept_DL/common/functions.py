# 阶跃函数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=int)


def Relu(x):
    return np.maximum(0, x)


def identity(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_matrix(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    else:
        x = x - np.max(x)


# 损失函数
# Mean Squared Error,y是预测值，t是当前的真实值
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交叉熵误差Cross Entropy Error ti是正确解标签，并且只有:当ti是正确解的时候标签值才为1，其余一直是0 ~~one-hot
def cross_entropy_error(y, t):
    # 将y转换为二维
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 将t(one-hot)独热编码转换为顺序编码
    if t.size == y.size:
        t = t.argmax(axis=1)
    n = y.shape[0]
    return -np.sum(np.log(y[np.arange(n), t] + 1e-10)) / n


if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
    print(step_function(x))
    print(Relu(x))
    print(softmax(x))
    x = np.array([[0, 1, 2, ], [3, 4, 5], [6, 7, 8], [-1, -2, -3]])
    print(softmax_matrix(x))
