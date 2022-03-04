import sympy
import random
import numpy as np

#%%%
# Question 3
epsilon = 0.05
delta = 0.05
d_vc = 10
N = [420000, 440000, 460000, 480000, 500000]
for n in N:
    e_square = 8 / n * np.log(4 * (2 * n) ** d_vc / delta)
    print("N:", n)
    print("Estimated e Square:", e_square)
    print("Error:", abs(epsilon ** 2 - e_square))
    print("----------------------------------------")

#%%%
# Question 4 and 5
d_vc = 50
delta = 0.05
N = 10000
N = 5
epsilon = sympy.symbols('epsilon')

print("Original VC Bound:", np.sqrt(8 / N * np.log(4 * (2 * N) ** d_vc / delta)))
print("Rademacher Penalty Bound:", (np.sqrt(2 * np.log(2.0 * N * N ** d_vc) / N) + (np.sqrt(2 / N * np.log(1 / delta)))))
print("Parrondo and Van Den Broek Bound:", sympy.solve(epsilon - sympy.sqrt(1 / N * (2 * epsilon + sympy.ln(6 * (2 * N) ** d_vc / delta)))))
print("Devroye:", sympy.solve(epsilon - sympy.sqrt(
    1 / (2 * N) * (4 * epsilon * (1 + epsilon) + np.log(4) + d_vc * np.log(N ** 2) - np.log(delta))
)))

print("Variant VC Bound:", np.sqrt(16 / N * np.log(2 * N ** d_vc / np.sqrt(delta))))


#%%%
# Question 17
import random
import numpy as np

def generate_data(size=20, noise=0.2):
    xs = []
    ys = []
    for i in range(size):
        x = random.uniform(-1, 1)
        prob = random.uniform(0, 1)
        if prob < noise:
            y = -np.sign(x)
        else:
            y = np.sign(x)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def hFunc(x, s, theta):
    if x - theta == 0:
        return s
    else:
        return np.sign(x - theta) * s

def compute_E_in(x, y, s, theta):
    errorNum = 0
    for i in range(len(x)):
        if y[i] != hFunc(x[i], s, theta):
            errorNum += 1
    return errorNum / len(x)

def trainDecisionStump():
    x, y = generate_data()
    best_E_in = 1
    sorted_x = np.sort(x)
    for current_theta in sorted_x:
        for s in [-1, 1]:
            current_E_in = compute_E_in(x, y, s, current_theta)
            if current_E_in < best_E_in:
                best_E_in = current_E_in
    return best_E_in


iteration = 5000
err_in_sum = 0
for i in range(iteration):
    err_in = trainDecisionStump()
    err_in_sum += err_in
    if i % 1000 == 999:
        print("iteration: ", i + 1)
print("total errorRate in sample is", err_in_sum / iteration)
