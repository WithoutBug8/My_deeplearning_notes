import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    # 将数据都转换成ndarray
    y_train = y_train.values
    y_test = y_test.values

    return x_test, y_test, x_train, y_train