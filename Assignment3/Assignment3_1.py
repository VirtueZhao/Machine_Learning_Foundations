import numpy as np
from sympy import symbols, diff, exp
#%%%
# Question 6
u, v = symbols('u v', real=True)
partial_diff_u = diff(exp(u) + exp(2 * v) + exp(u * v) + u**2 - 2 * u * v + 2 * v**2 - 3 * u - 2 * v, u)
partial_diff_v = diff(exp(u) + exp(2 * v) + exp(u * v) + u**2 - 2 * u * v + 2 * v**2 - 3 * u - 2 * v, v)
print("Partial u:", partial_diff_u)
print("Partial v:", partial_diff_v)
print("Gradient u:", partial_diff_u.subs({u: 0, v: 0}))
print("Gradient v:", partial_diff_v.subs({u: 0, v: 0}))

#%%%
# Question 7
u, v = symbols('u v', real=True)
u_val = 0
v_val = 0
eta = 0.01
print("u, v:", [u_val, v_val])
partial_diff_u = diff(exp(u) + exp(2 * v) + exp(u * v) + u ** 2 - 2 * u * v + 2 * v ** 2 - 3 * u - 2 * v, u)
partial_diff_v = diff(exp(u) + exp(2 * v) + exp(u * v) + u ** 2 - 2 * u * v + 2 * v ** 2 - 3 * u - 2 * v, v)

for i in range(5):
    print("Iteration:", i)
    u_val = u_val - eta * partial_diff_u.subs({u: u_val, v: v_val})
    v_val = v_val - eta * partial_diff_v.subs({u: u_val, v: v_val})
    print("u, v:", [u_val, v_val])
    E = exp(u_val) + exp(2 * v_val) + exp(u_val * v_val) + u_val ** 2 - 2 * u_val * v_val + 2 * v_val ** 2 - 3 * u_val - 2 * v_val
    print("E:", E)

#%%%
# Question 8
u, v = symbols('u v', real=True)
F = exp(u) + exp(2 * v) + exp(u * v) + u ** 2 - 2 * u * v + 2 * v ** 2 - 3 * u - 2 * v
print("f:", F.subs({u:0, v:0}))
F_u = diff(F, u)
F_v = diff(F, v)
print("f_u:", F_u.subs({u:0, v:0}))
print("f_v:", F_v.subs({u:0, v:0}))

F_uv = diff(F_u, v)
F_vu = diff(F_v, u)
print("f_uv:", F_uv.subs({u:0, v:0}) / 2)
print("f_vu:", F_vu.subs({u:0, v:0}) / 2)

F_uu = diff(F_u, u)
F_vv = diff(F_v, v)
print("f_uu:", F_uu.subs({u:0, v:0}) / 2)
print("f_vv:", F_vv.subs({u:0, v:0}) / 2)

#%%%
# Question 10
u, v = symbols('u v', real=True)
F = exp(u) + exp(2 * v) + exp(u * v) + u ** 2 - 2 * u * v + 2 * v ** 2 - 3 * u - 2 * v
F_u = diff(F, u)
F_v = diff(F, v)
F_uv = diff(F_u, v)
F_vu = diff(F_v, u)
F_uu = diff(F_u, u)
F_vv = diff(F_v, v)

u_val = 0
v_val = 0
for i in range(5):
    print("Iteration:", i)
    F_u_val = float(F_u.subs({u: u_val, v: v_val}))
    F_v_val = float(F_v.subs({u: u_val, v: v_val}))
    F_uu_val = float(F_uu.subs({u: u_val, v: v_val}))
    F_uv_val = float(F_uv.subs({u: u_val, v: v_val}))
    F_vu_val = float(F_vu.subs({u: u_val, v: v_val}))
    F_vv_val = float(F_vv.subs({u: u_val, v: v_val}))
    hession_matrix = np.mat([[F_uu_val, F_uv_val], [F_uv_val, F_vv_val]])
    grad = np.array([[F_u_val], [F_v_val]])
    delta = (hession_matrix.I * grad).tolist()
    u_val = u_val - delta[0][0]
    v_val = v_val - delta[1][0]
    E = exp(u_val) + exp(2 * v_val) + exp(u_val * v_val) + u_val ** 2 - 2 * u_val * v_val + 2 * v_val ** 2 - 3 * u_val - 2 * v_val
    print("E:", E)


#%%%
# Question 13
import numpy as np
import random

def generateData(size=1000):
    xs = []
    ys = []
    for i in range(size):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        y = np.sign(x1 ** 2 + x2 ** 2 - 0.6)
        prob = random.uniform(0, 1)
        if prob < 0.1:
            y = -y
        xs.append([1, x1, x2])
        ys.append([y])
    return np.mat(xs), np.array(ys)


def errorRate(w, x, y):
    pred_y = np.array(x * w)
    print(y)
    # print(pred_y)
    pred_y = np.array(list(map(np.sign, pred_y)))
    errorNum = np.sum(pred_y != y)
    return errorNum / len(y)

errorSum = 0
for i in range(1):
    print("Iteration:", i)
    x, y = generateData()
    w = np.array(np.linalg.pinv(x) * y)
    errorSum += errorRate(w, x, y)

print("Average In Sample Error:", errorSum/1000)

#%%%
# Question 14 & 15
import numpy as np
import random

def generateData(size=1000):
    xs = []
    ys = []
    for i in range(size):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        y = np.sign(x1 ** 2 + x2 ** 2 - 0.6)
        prob = random.uniform(0, 1)
        if prob < 0.1:
            y = -y
        xs.append([1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
        ys.append([y])
    return np.mat(xs), np.array(ys)


def errorRate(w, x, y):
    pred_y = np.array(x * w)
    pred_y = np.array(list(map(np.sign, pred_y)))
    errorNum = np.sum(pred_y != y)
    return errorNum / len(y)


errorSum = 0
for i in range(1000):
    if (i + 1) % 100 == 0:
        print("Iteration:", i+1)
    x, y = generateData()
    # w = np.array(np.linalg.pinv(x) * y)
    # w = np.array([[-1], [-0.05], [0.08], [0.13], [1.5], [15]])
    # w = np.array([[-1], [-0.05], [0.08], [0.13], [15], [1.5]])
    # w = np.array([[-1], [-1.5], [0.08], [0.13], [0.05], [1.5]])
    # w = np.array([[-1], [-0.05], [0.08], [0.13], [1.5], [1.5]])
    w = np.array([[-1], [-1.5], [0.08], [0.13], [0.05], [0.05]])
    errorSum += errorRate(w, x, y)

print("Average In Sample Error:", errorSum/1000)
