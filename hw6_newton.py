import numpy as np
import pandas as pd

iteration = 0
res = pd.DataFrame([], columns=["iteration", "x", "f(x)", "f'(x)"])
x = 1
epsilon = 1e-6

while True:
    f = np.exp(x) + np.exp(-x)
    f_p1 = np.exp(x) - np.exp(-x)
    f_p2 = f

    res.loc[iteration] = [iteration, x, f, f_p1]
    if abs(f_p1) < epsilon:
        break
    delta_x = -f_p1 / f_p2
    x += delta_x
    iteration += 1
res.to_csv("hw6_newton.csv", index=False)
print(res.to_string(index=False))
