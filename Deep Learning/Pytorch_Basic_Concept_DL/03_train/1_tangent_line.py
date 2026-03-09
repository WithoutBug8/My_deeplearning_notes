import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_diff


# 假设原函数是 y = 0.01x^2 + 0.1x
def f(x):
    return 0.01 * x** 2 + 0.1 * x


# 切线方程函数,返回切线函数 y = ax + b
def tangent_line(f, x):
    y = f(x)
    # 计算x处切线的斜率,利用数值微分计算斜率
    a = numerical_diff(f, x)
    print("切线斜率为:",a)
    # 根据切线过（x,y）点，计算截距
    b = y - a * x
    return lambda x: a * x + b


# 定义画图的范围
x = np.arange(0.0, 20.0, 0.1)
y = f(x)

# 计算某一处x的切线方程
f_line = tangent_line(f, x)
y_line = f_line(x)
# 原函数曲线
plt.plot(x, y)
# 切线
plt.plot(x, y_line)
plt.show()