import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing

def load_data(data_file, attributes=7, sep=','):
    #data = pd.read_table(data_file, sep=sep)
    data = pd.read_csv(data_file, sep=sep)
    le = preprocessing.LabelEncoder()
    for i in range(attributes):
        data.iloc[:,i] = le.fit_transform(data.iloc[:,i])
    x = data.values[:, :-1]
    y = data.values[:, -1]
    return x, y

def load_data2(data_file, attributes=23, sep=','):
    data = pd.read_csv(data_file, sep=sep)
    le = preprocessing.LabelEncoder()
    for i in range(attributes):
        data.iloc[:,i] = le.fit_transform(data.iloc[:,i])
    y = data.values[:, 0]
    x = data.values[:, 1:]
    return x, y

def split_train_test(x, y, test_size=0.33):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__=='__main__':
    x, y = load_data('car.data')
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    print('x_train: {} \ny_train: {} \nx_test: {} \ny_test: {}'.format(x_train, y_train, x_test, y_test))
