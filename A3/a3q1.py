import numpy as np
import scipy as sp
from scipy import linalg as spl
from matplotlib import pyplot as plt
import time as t


def f(x, A, I):
    return x + x @ (I - A @ x)


def Q1b(A, x0, I):
    xs = [x0]
    curr_x = x0
    new_x = f(curr_x, A, I)
    while not np.isclose(np.linalg.norm(curr_x - new_x), 0):
        xs.append(new_x)
        curr_x = new_x
        new_x = f(curr_x, A, I)
    return new_x


def random_matrix_generator(n):
    A = None
    while True:
        A = np.random.uniform(0, 100, size=(n, n))
        if np.linalg.matrix_rank(A) == n:
            break
    return A


if __name__ == "__main__":
    error1 = []
    error2 = []
    runtime1 = []
    runtime2 = []
    domain = []
    for k in range(1, 11):
        i = 2**k
        domain.append(i)
        matrix = random_matrix_generator(i)
        x0 = matrix.T / (np.linalg.norm(matrix, ord=1) * np.linalg.norm(matrix, ord=np.inf))
        I = np.identity(i)
        a = t.perf_counter()
        library_inv = spl.inv(matrix)
        b = t.perf_counter()
        error1.append(np.linalg.norm(I - matrix @ library_inv))
        runtime1.append((b - a) * 1000)

        a1 = t.perf_counter()
        newton_inv = Q1b(matrix, x0, I)
        b1 = t.perf_counter()
        error2.append(np.linalg.norm(I - matrix @ newton_inv))
        runtime2.append((b1 - a1) * 1000)

    # plt.plot(domain, error1, label="scipy.linalg.inv()")
    # plt.plot(domain, error2, label="Newton's method")
    # plt.xlabel("Size of matrix")
    # plt.ylabel("Norm of residual matrix")
    # plt.title("Norm of the residual matrix vs matrix size")
    # plt.legend()
    # plt.savefig("Q1_error.png")

    plt.plot(domain, runtime1, label="scipy.linalg.inv()")
    plt.plot(domain, runtime2, label="Newton's method")
    plt.title("Runtime of methods vs size of matrix")
    plt.xlabel("Size of matrix")
    plt.ylabel("Runtime in ms")
    plt.legend()
    plt.savefig("Q1_runtime.png")
    plt.show()
