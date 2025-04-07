import numpy as np
from scipy.optimize import minimize_scalar
# f(x)=(10x1^2+x2^2)/2
# f'(x)=10x1+x2


def f(x1, x2):
    return (10 * x1**2 + x2**2) / 2


def grad_f(x1, x2):
    return 10 * x1 + x2


def phi(z, point):
    arr = point - z * grad_f(*point)
    return f(*arr)


def steepest_decrease(start, err):
    x = start.copy()
    grad = grad_f(*x)
    grad_norm = np.linalg.norm(grad)
    iteration = 0
    while grad_norm >= err:
        iteration += 1
        res = minimize_scalar(phi, args=(x,))
        x = x - res.x * x
        grad = grad_f(*x)
        grad_norm = np.linalg.norm(grad)
    return x, iteration


if __name__ == "__main__":
    err = 1e-6
    start = np.array([0.2, 1])
    x, iteration = steepest_decrease(start, err)
    print("using steepest decrease method")
    print(iteration, x, f(*x), grad_f(*x), np.linalg.norm(grad_f(*x)), sep="\n")
