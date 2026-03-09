import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from common.optimizer import *


# 定义目标函数，f(x,y) = 1/20 * x^2 + y^2
def f(x, y):
    return x ** 2 / 20 + y ** 2


# 定义目标函数的梯度,得到一个长度为2的向量
def f_grad(x, y):
    return x / 10, 2 * y


# 定义初始点的位置
init_pos = (-7.0, 2.0)

# 定义参数和梯度
params = {}
grads = {}

# 定义优化器,超参数指定学习率
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.9)
optimizers["Momentum"] = Momentum(lr=0.11, momentum=0.85)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.5,alpha1=0.5)

# 子图序号
idx = 1
# 遍历优化器，用优化器更新参数求最小值点
for key in optimizers:
    optimizer = optimizers[key]
    # 记录参数点更新的历史
    x_history = []
    y_history = []
    # 参数初始化
    params['x'], params['y'] = init_pos[0], init_pos[1]
    # 指定迭代30次
    for i in range(30):
        # 保存当前点坐标
        x_history.append(params['x'])
        y_history.append(params['y'])
        # 1. 计算梯度
        grads['x'], grads['y'] = f_grad(params['x'], params['y'])
        # 2. 更新参数
        optimizer.update(params, grads)
    # 画图
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    Z[Z > 7] = 0  # Z高度大于7就不画图了
    # 定义子图
    plt.subplot(2, 2, idx)
    idx += 1
    # 绘制等高线
    plt.contour(X, Y, Z)
    # 绘制最小值点
    plt.plot(0, 0, '+')
    # 绘制轨迹曲线
    plt.plot(x_history, y_history, 'o-', color='red', label=key, markersize=2)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.legend(loc='best')

plt.show()
