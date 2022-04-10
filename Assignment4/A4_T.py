import math
import matplotlib.pyplot as plt
import numpy as np
import re

#%%%
import pandas as pd


def sign(x):
    if x > 0:
        return +1
    else: return -1


def visualize(data,W=[]):
    nx = []
    ny = []
    ox = []
    oy = []
    for i in range(len(data)):
        if data[i][-1] == -1:
            nx.append(data[i][0])
            ny.append(data[i][1])
        else:
            ox.append(data[i][0])
            oy.append(data[i][1])
    plt.scatter(nx,ny,marker="x",c="r")
    plt.scatter(ox,oy,marker="o",c="g")
    if len(W)!=0 :
        x = np.linspace(0, 1, 50)
        y = -W[1] / W[2] * x - W[0] / W[2]
        plt.plot(x, y, color="black")
    plt.show()
def ridge_regression_one_step(data,_lambda):
    X_matrix = []
    Y_matrix = []
    for i in range(len(data)):
        temp = [1]
        for j in range(len(data[i])-1):
            temp.append(data[i][j])

        X_matrix.append(temp)
        Y_matrix.append([data[i][-1]])
    X = np.mat(X_matrix)
    hatX = math.sqrt(_lambda)*np.eye(len(data[0]))
    hatY = np.mat([ 0 for i in data[0]]).T
    Y = np.mat(Y_matrix)

    W = (X.T*X + hatX.T*hatX).I*(X.T*Y+hatX.T*hatY)
    return W.T.tolist()[0]

def Ein(data,W):
    err_num = 0
    for i in range(len(data)):
        res = W[0]
        for j in range(1,len(W)):
            res += W[j]*data[i][j-1]

        if sign(res) != data[i][-1]:
            err_num+=1
    return err_num

def readDataFrom(path):
    separator = re.compile('\t|\b| |\n')
    result = []
    with open(path,"r") as f:
        s = f.readline()[:-1]
        while s:
            temp = separator.split(s)
            result.append([float(x) for x in temp])
            s = f.readline()[:-1]
    return result


# def ridge_regression(data, _lambda):
#     data = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
#
#     x0 = np.ones(len(data[data.columns[0]]))
#     data.insert(loc=0, column='x0', value=x0)
#
#     X = np.matrix(data[data.columns[:-1]])
#     Y = np.matrix(data[data.columns[-1]]).T
#     X_hat = math.sqrt(_lambda) * np.eye(X.shape[1])
#     Y_hat = np.mat(np.zeros(X_hat.shape[0]).reshape(X_hat.shape[0], 1))
#
#     w = (X.T * X + X_hat.T * X_hat).I * (X.T * Y + X_hat.T * Y_hat)
#     return w.T.tolist()[0]

def cv(data,fold_count,_lambda):
    # disorder data
    ecv = 0
    each_c = len(data)/fold_count
    for i in range(fold_count):
        val = data[int(i*each_c):int((i+1)*each_c)]
        train = data[0:int(i*each_c)]
        train.extend(data[int((i+1)*each_c):-1])
        # W = ridge_regression(train,_lambda)
        W = ridge_regression_one_step(train, _lambda)
        ecv +=Ein(val,W)/len(val)
        print("Fold:", i)
        print("w:", W)
        # print(val)

    return ecv/fold_count


if __name__ == "__main__":
    minEtrain = 1
    minEval = 1
    minEout = 1
    minEvalI = -1
    minEtrainI = -1
    minEoutI = -1
    for i in range(-10,3):
        _lambda = math.pow(10,i)
        data = readDataFrom("hw4_train.dat")
        data_test = readDataFrom("hw4_test.dat")
        E_cv =cv(data, 5, _lambda)
        print(i)
        print("E_cv:", E_cv)
        break