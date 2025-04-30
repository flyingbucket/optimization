import numpy as np
# f(x)=(10x1^2+x2^2)/2


def f(x1, x2):
    return (10 * x1**2 + x2**2) / 2


def grad_f(x1, x2):
    return np.array((10 * x1, x2))


H = np.array([[10, 0], [0, 1]])


def newton(start, err):
    x = start.copy()
    grad = grad_f(*x)
    grad_norm = np.linalg.norm(grad)
    iteration = 0
    while grad_norm**2 >= err:
        iteration += 1
        z = (grad @ grad) / (grad @ H @ grad)
        x = x - z * grad
        grad = grad_f(*x)
        grad_norm = np.linalg.norm(grad)
    return x, iteration


if __name__ == "__main__":
    err = 1e-10
    start = np.array([0.2, 1])
    x, iteration = newton(start, err)
    print("using steepest decrease method")
    print(
        f"iteration:{iteration}",
        f"x:{x}",
        f"f(*x):{f(*x)}",
        f"grad_f(*x):{grad_f(*x)}",
        f"grad_norm:{np.linalg.norm(grad_f(*x))}",
        sep="\n",
    )
