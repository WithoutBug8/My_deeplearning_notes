import numpy as np
# 数值微分求导
def numerical_diff(f, x):
    # 取一个很小的值h,不要给的太小会影响精度
    h = 1e-4
    return (f(x + h) - f(x)) / h


# 中心差分，x不仅要+h，还要-h，最后取一半的值
def numerial_diff_central(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

# 数值微分求梯度,传入的x是向量
def _numerial_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # 遍历x中的特征xi
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * h)
        # 恢复xi的值，给其他维度用
        x[i] = tmp
    return grad

# 传入的X是矩阵
def numerial_gradient(f, X):
    if X.ndim == 1:
        return _numerial_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        # 遍历X中的每一行数据，分别求梯度
        for i,x in enumerate(X):
            grad[i] = _numerial_gradient(f,x)
        return grad