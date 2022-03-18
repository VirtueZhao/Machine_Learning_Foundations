import pandas as pd
import numpy as np


#%%%
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

# def StochasticGradientDescent(w, train_X, train_Y, i):
#     gradient = sigmoid(-train_Y[i] * train_X[i] * w).item() * -train_Y[i] * train_X[i]
#     gradient = gradient.reshape(-1, 1)
#
#     return gradient

def GradientDescent(w, train_X, train_Y):
    gradient_total = 0
    for i in range(len(train_Y)):
        gradient = sigmoid(-train_Y[i] * train_X[i] * w).item() * -train_Y[i] * train_X[i]
        gradient_total += gradient.reshape(-1, 1)
    # print(gradient_total)
    return gradient_total / len(train_Y)



def logisticRegression(train_X, train_Y):
    eta = 0.001
    # eta = 0.01
    T = 2000
    w = np.zeros(train_X[0].size)
    w = w.reshape(-1, 1)
    for i in range(T):
        # gradient = StochasticGradientDescent(w, train_X, train_Y, i % len(train_Y))
        gradient = GradientDescent(w, train_X, train_Y)
        w = w - eta * gradient
        if (i+1) % 100 == 0:
            print("Iteration:", i+1)
    return w


def errorRate(w, test_X, test_Y):
    pred_y = (sigmoid(test_X * w)).tolist()
    for i in range(len(pred_y)):
        if pred_y[i][0] > 0.5:
            pred_y[i] = 1
        else:
            pred_y[i] = -1
    errorNum = np.sum(np.array(pred_y) != np.array(test_Y))
    return errorNum / len(test_Y)


train_data = pd.read_csv('hw3_train.dat', header=None, delimiter=r'\s+')
train_data.columns = train_data.columns.astype(str)
x0_train = np.ones(len(train_data['20']))
train_data.insert(loc=0, column='x0', value=x0_train)
test_data = pd.read_csv('hw3_test.dat', header=None, delimiter=r'\s+')
test_data.columns = test_data.columns.astype(str)
x0_test = np.ones(len(test_data['20']))
test_data.insert(loc=0, column='x0', value=x0_test)
test_X = test_data.drop(labels='20', axis=1)
test_Y = test_data['20']
train_X = train_data.drop(labels='20', axis=1)
train_Y = train_data['20']
train_X = np.mat(train_X.values.tolist())
test_X = np.mat(test_X.values.tolist())

w = logisticRegression(train_X, train_Y)
# print(w)
error = errorRate(w, test_X, test_Y)
print("Out of Sample Error:", error)









