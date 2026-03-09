import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax


# 1. 读取数据
def get_data():
    # 1.get the data from dataset
    data = pd.read_csv('../Datasets/digit-recognizer/train.csv')
    # 2.split the dataset
    x = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # 3.特征工程，归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test, y_test


# 2.创建模型
def init_network():
    # 直接从以下文件中加载字典对象——训练好的参数放在这个文件里
    network = joblib.load("../Datasets/digit-recognizer/nn_sample")
    return network


# 3. 前向传播
def forward_propagation(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 逐层计算
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y


# 主流程
# 1. Get the test data
x, y = get_data()
print(x.shape)
print(y.shape)
# 2. get the model parameter
network = init_network()
# 查看各个层数的神经元，输入输出 数据
print(network['W1'].shape)
print(network['W2'].shape)
print(network['W3'].shape)
print(network['b1'].shape)
print(network['b2'].shape)
print(network['b3'].shape)

# 定义变量
batch_size = 100
accuracy_count = 0
n = x.shape[0]
# 3. 循环迭代，分批次测试，前向传播，并累计预测准确个数
for i in range(0, n, batch_size):
    # 3.1 取出当前批次的数据
    x_batch = x[i:i + batch_size]
    # 3.2 前向传播
    y_batch = forward_propagation(network, x_batch)
    # 3.3 将输出分类概率转换为分类标签
    y_pred = np.argmax(y_batch, axis=1)
    # 3.4 累加准确个数
    accuracy_count += np.sum(y_pred == y[i:i + batch_size])
# 4.计算准确率
print("Accuracy count: ", accuracy_count / n)
