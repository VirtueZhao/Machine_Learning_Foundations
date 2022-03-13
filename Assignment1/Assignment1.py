import math

# Question 9
print(math.factorial(10) / math.factorial(5) / math.factorial(5) * 0.5**10)
# Question 10
print(math.factorial(10) / math.factorial(9) * 0.9 ** 9 * 0.1)
# Question 11
print(0.1 ** 10 + math.factorial(10) / math.factorial(9) * 0.1 ** 9 * 0.9)
# Question 12
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
#%%%
import copy as cp
import numpy as np
import pandas as pd
# Question 18
def get_training_set():
    train_data = pd.read_csv("Assignment1/hw1_18_train.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    train_size = train_data.shape[0]
    X_train = np.array(train_data.iloc[:, 0:4])
    X_train = np.hstack((np.ones((train_size, 1)), X_train))
    y_train = np.array(train_data.iloc[:, 4:5])
    permutation_index = np.random.permutation(train_size)
    # print(list)
    X_train = X_train[permutation_index]
    y_train = y_train[permutation_index]
    return X_train, y_train, train_size

def get_test_set():
    test_data = pd.read_csv("Assignment1/hw1_18_test.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    test_set = np.array(test_data)
    test_size = test_set.shape[0]
    x_test = test_set[:, 0:4]
    x_test = np.hstack((np.ones((test_size, 1)), x_test))
    y_test = test_set[:, 4:5]
    return x_test, y_test, test_size

def Pocket():
    X_train, y_train, train_size = get_training_set()

    w = np.zeros((5, 1))
    w_pocket = np.zeros((5, 1))
    learning_rate = 0.5
    max_allowed_mistake = train_size
    update_count = 0

    for i in range(train_size):
        data_x = X_train[i, :]
        data_y = y_train[i, :]
        loss = np.dot(data_x, w) * data_y
        if loss <= 0:
            w += learning_rate * data_y * data_x.reshape(5, 1)
            update_count += 1
            current_w_mistake = 0
            for j in range(train_size):
                val_data_x = X_train[j, :]
                val_data_y = y_train[j, :]
                val_loss = np.dot(val_data_x, w) * val_data_y
                if val_loss <= 0:
                    current_w_mistake += 1
            if current_w_mistake < max_allowed_mistake:
                max_allowed_mistake = current_w_mistake
                w_pocket = cp.deepcopy(w)

            if update_count == 50:
                break
    return w_pocket

def Pocket_Error(w_pocket):
    X_test, y_test, test_size = get_test_set()
    error_num = 0

    for i in range(test_size):
        data_x = X_test[i, :]
        data_y = y_test[i, :]
        loss = np.dot(data_x, w_pocket) * data_y
        if loss <= 0:
            error_num += 1
    error_ratio = error_num / test_size

    return error_ratio


error_ratio_sum = 0
for i in range(2000):
    print("Iteration:", str(i))
    w_pocket = Pocket()
    error_ratio = Pocket_Error(w_pocket)
    print("Error Ratio:", error_ratio)
    error_ratio_sum += error_ratio


ave_error_ratio = error_ratio_sum / 2000
print(ave_error_ratio)

#%%%
import copy as cp
import numpy as np
import pandas as pd
# Question 19
def get_training_set():
    train_data = pd.read_csv("Assignment1/hw1_18_train.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    train_size = train_data.shape[0]
    X_train = np.array(train_data.iloc[:, 0:4])
    X_train = np.hstack((np.ones((train_size, 1)), X_train))
    y_train = np.array(train_data.iloc[:, 4:5])
    permutation_index = np.random.permutation(train_size)
    # print(list)
    X_train = X_train[permutation_index]
    y_train = y_train[permutation_index]
    return X_train, y_train, train_size

def get_test_set():
    test_data = pd.read_csv("Assignment1/hw1_18_test.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    test_set = np.array(test_data)
    test_size = test_set.shape[0]
    x_test = test_set[:, 0:4]
    x_test = np.hstack((np.ones((test_size, 1)), x_test))
    y_test = test_set[:, 4:5]
    return x_test, y_test, test_size

def Pocket():
    X_train, y_train, train_size = get_training_set()

    w = np.zeros((5, 1))
    w_pocket = np.zeros((5, 1))
    learning_rate = 0.5
    max_allowed_mistake = train_size
    update_count = 0

    for i in range(train_size):
        data_x = X_train[i, :]
        data_y = y_train[i, :]
        loss = np.dot(data_x, w) * data_y
        if loss <= 0:
            w += learning_rate * data_y * data_x.reshape(5, 1)
            update_count += 1
            current_w_mistake = 0
            for j in range(train_size):
                val_data_x = X_train[j, :]
                val_data_y = y_train[j, :]
                val_loss = np.dot(val_data_x, w) * val_data_y
                if val_loss <= 0:
                    current_w_mistake += 1
            if current_w_mistake < max_allowed_mistake:
                max_allowed_mistake = current_w_mistake
                w_pocket = cp.deepcopy(w)

            if update_count == 50:
                break
    return w

def Pocket_Error(w_pocket):
    X_test, y_test, test_size = get_test_set()
    error_num = 0

    for i in range(test_size):
        data_x = X_test[i, :]
        data_y = y_test[i, :]
        loss = np.dot(data_x, w_pocket) * data_y
        if loss <= 0:
            error_num += 1
    error_ratio = error_num / test_size

    return error_ratio


error_ratio_sum = 0
for i in range(2000):
    print("Iteration:", str(i))
    w_pocket = Pocket()
    error_ratio = Pocket_Error(w_pocket)
    print("Error Ratio:", error_ratio)
    error_ratio_sum += error_ratio


ave_error_ratio = error_ratio_sum / 2000
print(ave_error_ratio)

#%%%
import copy as cp
import numpy as np
import pandas as pd
# Question 20
def get_training_set():
    train_data = pd.read_csv("Assignment1/hw1_18_train.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    train_size = train_data.shape[0]
    X_train = np.array(train_data.iloc[:, 0:4])
    X_train = np.hstack((np.ones((train_size, 1)), X_train))
    y_train = np.array(train_data.iloc[:, 4:5])
    permutation_index = np.random.permutation(train_size)
    # print(list)
    X_train = X_train[permutation_index]
    y_train = y_train[permutation_index]
    return X_train, y_train, train_size

def get_test_set():
    test_data = pd.read_csv("Assignment1/hw1_18_test.dat", sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'y'])
    test_set = np.array(test_data)
    test_size = test_set.shape[0]
    x_test = test_set[:, 0:4]
    x_test = np.hstack((np.ones((test_size, 1)), x_test))
    y_test = test_set[:, 4:5]
    return x_test, y_test, test_size

def Pocket():
    X_train, y_train, train_size = get_training_set()

    w = np.zeros((5, 1))
    w_pocket = np.zeros((5, 1))
    learning_rate = 0.5
    max_allowed_mistake = train_size
    update_count = 0

    for i in range(train_size):
        data_x = X_train[i, :]
        data_y = y_train[i, :]
        loss = np.dot(data_x, w) * data_y
        if loss <= 0:
            w += learning_rate * data_y * data_x.reshape(5, 1)
            update_count += 1
            current_w_mistake = 0
            for j in range(train_size):
                val_data_x = X_train[j, :]
                val_data_y = y_train[j, :]
                val_loss = np.dot(val_data_x, w) * val_data_y
                if val_loss <= 0:
                    current_w_mistake += 1
            if current_w_mistake < max_allowed_mistake:
                max_allowed_mistake = current_w_mistake
                w_pocket = cp.deepcopy(w)

            if update_count == 100:
                break
    return w_pocket

def Pocket_Error(w_pocket):
    X_test, y_test, test_size = get_test_set()
    error_num = 0

    for i in range(test_size):
        data_x = X_test[i, :]
        data_y = y_test[i, :]
        loss = np.dot(data_x, w_pocket) * data_y
        if loss <= 0:
            error_num += 1
    error_ratio = error_num / test_size

    return error_ratio


error_ratio_sum = 0
for i in range(2000):
    print("Iteration:", str(i))
    w_pocket = Pocket()
    error_ratio = Pocket_Error(w_pocket)
    print("Error Ratio:", error_ratio)
    error_ratio_sum += error_ratio


ave_error_ratio = error_ratio_sum / 2000
print(ave_error_ratio)