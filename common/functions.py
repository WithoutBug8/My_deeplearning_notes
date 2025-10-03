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
        x = x - np.max(x, axis = 0)
        y = np.exp(x) / np.sum(np.exp(x) , axis = 0)
        return y.T
    else:
        x = x - np.max(x)



if __name__ == '__main__':
    x = np.array([0,1,2,3,4,5,-1,-2,-3,-4,-5])
    print(step_function(x))
    print(Relu(x))
    print(softmax(x))
    x = np.array([[0,1,2,],[3,4,5],[6,7,8],[-1,-2,-3]])
    print(softmax_matrix(x))