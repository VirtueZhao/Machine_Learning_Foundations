import math


print(math.factorial(10) / math.factorial(5) / math.factorial(5) * 0.5**10)
print(math.factorial(10) / math.factorial(9) * 0.9 ** 9 * 0.1)
print(0.1 ** 10 + math.factorial(10) / math.factorial(9) * 0.1 ** 9 * 0.9)
print(2 * math.e ** (-2 * 0.8 ** 2 * 10))
#%%%
import pandas as pd
import numpy as np
# Question 15
def get_training_set():
    sample_data = pd.read_csv("Assignment1/hw1_15_train.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    sample_size = sample_data.shape[0]
    X_train = np.array(sample_data.iloc[:, 0:4])
    X_train = np.hstack((np.ones((sample_size, 1)), X_train))
    y_train = np.array(sample_data.iloc[:, 4:5])

    return X_train, y_train, sample_size

def PLA():
    X_train, y_train, sample_size = get_training_set()

    w = np.zeros((5, 1))
    num_iters = 0

    while True:
        Flag = True
        for i in range(sample_size):
            data_x = X_train[i, :]
            data_y = y_train[i, :]
            loss = np.dot(data_x, w) * data_y
            if loss <= 0:
                w += data_y * data_x.reshape(5, 1)
                num_iters += 1
                Flag = False
        if Flag:
            break
    return num_iters, w


num_iters, w = PLA()
print("Num of Iterations:", num_iters)
print("Final w:", w)
#%%%
import numpy as np
import pandas as pd
# Question 16
def get_training_set():
    sample_data = pd.read_csv("Assignment1/hw1_15_train.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    sample_size = sample_data.shape[0]
    X_train = np.array(sample_data.iloc[:, 0:4])
    X_train = np.hstack((np.ones((sample_size, 1)), X_train))
    y_train = np.array(sample_data.iloc[:, 4:5])

    permutation_index = np.random.permutation(sample_size)
    # print(list)
    X_train = X_train[permutation_index]
    y_train = y_train[permutation_index]

    return X_train, y_train, sample_size

def PLA():
    X_train, y_train, sample_size = get_training_set()

    w = np.zeros((5, 1))
    num_iters = 0

    while True:
        Flag = True
        for i in range(sample_size):
            data_x = X_train[i, :]
            data_y = y_train[i, :]
            loss = np.dot(data_x, w) * data_y
            if loss <= 0:
                w += data_y * data_x.reshape(5, 1)
                num_iters += 1
                Flag = False
        if Flag:
            break
    return num_iters, w


sum = 0
for i in range(2000):
    num_iters, w = PLA()
    sum += num_iters
print("Average Iter:", sum / 2000)
#%%%
import numpy as np
import pandas as pd
# Question 17
def get_training_set():
    sample_data = pd.read_csv("Assignment1/hw1_15_train.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    sample_size = sample_data.shape[0]
    X_train = np.array(sample_data.iloc[:, 0:4])
    X_train = np.hstack((np.ones((sample_size, 1)), X_train))
    y_train = np.array(sample_data.iloc[:, 4:5])

    permutation_index = np.random.permutation(sample_size)
    # print(list)
    X_train = X_train[permutation_index]
    y_train = y_train[permutation_index]

    return X_train, y_train, sample_size

def PLA():
    X_train, y_train, sample_size = get_training_set()

    w = np.zeros((5, 1))
    num_iters = 0
    learning_rate = 0.5

    while True:
        Flag = True
        for i in range(sample_size):
            data_x = X_train[i, :]
            data_y = y_train[i, :]
            loss = np.dot(data_x, w) * data_y
            if loss <= 0:
                w += learning_rate * data_y * data_x.reshape(5, 1)
                num_iters += 1
                Flag = False
        if Flag:
            break
    return num_iters, w


sum = 0
for i in range(2000):
    num_iters, w = PLA()
    sum += num_iters
print("Average Iter:", sum / 2000)
