import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

def random_matrix_generator(n):
    A = None
    B = None
    C = None
    while True:
        A = np.random.uniform(0, 100, size=(n, n))
        if np.linalg.matrix_rank(A) == n:
            break
    while True:
        B = np.random.uniform(0, 100, size=(n, n))
        if np.linalg.matrix_rank(B) == n:
            break
    while True:
        C = np.random.uniform(0, 100, size=(n, n))
        if np.linalg.matrix_rank(C) == n:
            break
    return A, B, C

def Q3_c():
    algorithm1 = []
    algorithm2 = []
    size = []
    for n in range(1, 1001):
        m_size = n*2
        size.append(m_size)
        b = np.ones((m_size,))
        A, B, C = random_matrix_generator(m_size)
        I = np.identity(m_size)


        s2 = time.perf_counter()
        y = sla.solve(C, b)
        righthand = (2 * A + I) @ (y + A @ b)
        result1 = sla.solve(B, righthand)
        e2 = time.perf_counter()
        algorithm2.append(e2 - s2)



        s1 = time.perf_counter()
        result2 = np.linalg.inv(B) @ (2*A + I) @ (np.linalg.inv(C) + A) @ b
        e1 = time.perf_counter()
        algorithm1.append(e1 - s1)

    return size, algorithm1, algorithm2


    plt.plot(size, algorithm1, label="Performance of algorithm 1")
    plt.plot(size, algorithm2, label="Performance of algorithm 2")
    plt.xlabel("Matrix size")
    plt.ylabel("Runtime")
    plt.title("Runtime of algorithm 1 vs algorithm 2")
    plt.legend()
    plt.savefig("3c.png")
    plt.show()

if __name__ =="__main__":
    Q3_c()