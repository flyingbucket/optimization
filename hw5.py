import numpy as np


def f(x):
    return np.exp(x) + np.exp(-x)


def golden_section_search(f, a, b, tol=0.1):
    phi = (np.sqrt(5) - 1) / 2  # 黄金比例常数 ≈ 0.618
    resphi = 1 - phi

    # 初始内部两个点
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    while (b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)

    # 返回最小值的估计点和对应的函数值
    x_min = (a + b) / 2
    return x_min, f(x_min), a, b


# 执行
x_min, f_min, a, b = golden_section_search(f, -0.5, 1.5, tol=0.1)
print(f"最小值点 x ≈ {x_min:.4f}, f(x) ≈ {f_min:.4f}")
print(f"停机区间 [{a},{b}]")
