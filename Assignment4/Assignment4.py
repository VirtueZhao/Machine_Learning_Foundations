#%%%
# Question 9
import numpy as np

p = 2 * np.e ** (-2 * 0.01 ** 2 * 10000)
print(p)
#%%%
# Question 13
import pandas as pd
import math
import numpy as np

def ridge_regression(data, _lambda):
    X = np.matrix(data[data.columns[:-1]])
    Y = np.matrix(data[data.columns[-1]]).T
    X_hat = math.sqrt(_lambda) * np.eye(X.shape[1])
    Y_hat = np.mat(np.zeros(X_hat.shape[0]).reshape(X_hat.shape[0], 1))

    w = (X.T * X + X_hat.T * X_hat).I * (X.T * Y + X_hat.T * Y_hat)
    return w.T.tolist()[0]


def error(data, w):
    err_num = 0
    X = data[data.columns[:-1]].to_numpy()
    Y = data[data.columns[-1]].to_numpy()
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        if np.sign(sum(w * x)) != y:
            err_num += 1
    return err_num / len(X)
#%%%

train_data = pd.read_csv("hw4_train.dat", sep='\s+', names=['x1', 'x2', 'y'])
x0 = np.ones(len(train_data[train_data.columns[0]]))
train_data.insert(loc=0, column='x0', value=x0)
test_data = pd.read_csv('hw4_test.dat', sep='\s+', names=['x1', 'x2', 'y'])
x0 = np.ones(len(test_data[test_data.columns[0]]))
test_data.insert(loc=0, column='x0', value=x0)

w = ridge_regression(train_data, 10)

E_in = error(train_data, w)
print("E_in:", E_in)
E_out = error(test_data, w)
print("E_out:", E_out)
#%%%
# Question 14, 15
for _lambda in range(-10, 3):
    _lambda = 10 ** _lambda
    print(_lambda)
    w = ridge_regression(train_data, _lambda)
    E_in = error(train_data, w)
    E_out = error(test_data, w)
    print("E_in:", E_in)
    print("E_out:", E_out)
    print("---")

#%%%
# Question 16, 17
train_data_all = pd.read_csv("hw4_train.dat", sep='\s+', names=['x1', 'x2', 'y'])
x0 = np.ones(len(train_data_all[train_data_all.columns[0]]))
train_data_all.insert(loc=0, column='x0', value=x0)
train_data = train_data_all.head(120)
train_data_val = train_data_all.tail(80)
test_data = pd.read_csv('hw4_test.dat', sep='\s+', names=['x1', 'x2', 'y'])
x0 = np.ones(len(test_data[test_data.columns[0]]))
test_data.insert(loc=0, column='x0', value=x0)

for _lambda in range(-10, 3):
    print("lambda:", _lambda)
    _lambda = 10 ** _lambda
    w = ridge_regression(train_data, _lambda)
    E_in = error(train_data, w)
    print("E_in:", E_in)
    E_val = error(train_data_val, w)
    print("E_val:", E_val)
    E_out = error(test_data, w)
    print("E_out:", E_out)
    print("---")

#%%%
# Question 18
_lambda = 10 ** 0
w = ridge_regression(train_data_all, _lambda)
E_in = error(train_data_all, w)
print("E_in:", E_in)
E_out = error(test_data, w)
print("E_out:", E_out)
#%%%
# Question 19, 20
import pandas as pd
import math
import numpy as np

def ridge_regression(data, _lambda):
    X = np.matrix(data[data.columns[:-1]])
    Y = np.matrix(data[data.columns[-1]]).T
    X_hat = math.sqrt(_lambda) * np.eye(X.shape[1])
    Y_hat = np.mat(np.zeros(X_hat.shape[0]).reshape(X_hat.shape[0], 1))

    w = (X.T * X + X_hat.T * X_hat).I * (X.T * Y + X_hat.T * Y_hat)
    return w.T.tolist()[0]


def error(data, w):
    err_num = 0
    X = data[data.columns[:-1]].to_numpy()
    Y = data[data.columns[-1]].to_numpy()
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        if np.sign(sum(w * x)) != y:
            err_num += 1
    return err_num / len(X)

train_data_all = pd.read_csv("hw4_train.dat", sep='\s+', names=['x1', 'x2', 'y'])
x0 = np.ones(len(train_data_all[train_data_all.columns[0]]))
train_data_all.insert(loc=0, column='x0', value=x0)
test_data = pd.read_csv('hw4_test.dat', sep='\s+', names=['x1', 'x2', 'y'])
x0 = np.ones(len(test_data[test_data.columns[0]]))
test_data.insert(loc=0, column='x0', value=x0)

for _lambda in range(-10, 3):
    print("lambda:", _lambda)
    _lambda = 10 ** _lambda
    E_cv = 0
    for i in range(5):
        val_data = train_data_all[40 * i: 40 * i + 40]
        train_data = train_data_all.copy()
        train_data = train_data.drop(labels=range(40 * i, 40 * i + 40))
        w = ridge_regression(train_data, _lambda)
        E_cv += error(val_data, w)
    w = ridge_regression(train_data_all, _lambda)
    E_in = error(train_data_all, w)
    E_out = error(test_data, w)
    print("E_cv:", E_cv / 5)
    print("E_in:", E_in)
    print("E_out:", E_out)