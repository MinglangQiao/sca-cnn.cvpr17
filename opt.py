# -*- coding: utf-8 -*-
import numpy as np

def get_Hierbelt(m=5):
    """
    opti homework, code by 樵明朗
    """
    x = 1. / (np.arange(1, m+1) + np.arange(0, m)[:, np.newaxis])
    print(x)

    return x

def conjugate_grad(A, b, x):
    """
    update 1028
    """
    n = len(b)
    # if not x:
        # x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-6:
            print('>>>> 循环次数:', (i+1))
            break
        p = beta * p - r
    return x

if __name__ == '__main__':

    n = 20
    A = get_Hierbelt(n)
    b = np.ones(n)
    x0 = np.zeros(n)
    # print(x0)
    # t

    print('start')
    x = conjugate_grad(A, b, x0)
    x_np = np.linalg.solve(A, b)
    print(">>>>: 自己实现结果： ", x)
    print(">>>>: 标准库的结果： ", x_np)
