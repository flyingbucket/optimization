import numpy as np
import pandas as pd

# y=e^x+e^(-x)
# y'=e^x-e^(-x)
# y''=e^x+e^(-x)

x = 1
t = abs(np.exp(x) - np.exp(-x))
res = pd.DataFrame([], columns=["x", "f(x)", "f'(x)"])

res.loc[0, "x"] = 1
res.loc[0, "f(x)"] = np.exp(x) + np.exp(-x)
res.loc[0, "f'(x)"] = np.exp(x) - np.exp(-x)

interation = 1
while t >= 10 ** (-6):
    delta_x = -(np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    x = x + delta_x
    x_1 = np.exp(x) - np.exp(-x)
    t = abs(x_1)
    res.loc[interation, "x"] = x
    res.loc[interation, "f(x)"] = np.exp(x) + np.exp(-x)
    res.loc[interation, "f'(x)"] = np.exp(x) - np.exp(-x)
    interation += 1

res = res.reset_index(names="iteration")
res.to_csv("hw6_newton.csv", index=False)
