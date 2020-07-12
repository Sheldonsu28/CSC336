#CSC336 Assignment #2 starter code

import numpy as np
import scipy.linalg as sla

#Q3c
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
        m_size = n * 2
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
        result2 = np.linalg.inv(B) @ (2 * A + I) @ (np.linalg.inv(C) + A) @ b
        e1 = time.perf_counter()
        algorithm1.append(e1 - s1)

    return size, algorithm1, algorithm2

#Q4a
def p_to_q(p):
    """
    return the permutation vector, q, corresponding to
    the pivot vector, p.
    >>> p_to_q(np.array([2,3,2,3]))
    array([2, 3, 0, 1])
    >>> p_to_q(np.array([2,4,8,3,9,7,6,8,9,9]))
    array([2, 4, 8, 3, 9, 7, 6, 0, 1, 5])
    """
    q = [i for i in range(0, p.shape[0])]
    for i in range(0, p.shape[0]):
        temp = q[i]
        q[i] = q[p[i]]
        q[p[i]] = temp
    return np.array(q)

#Q4b
def solve_plu(A,b):
    """
    return the solution of Ax=b. The solution is calculated
    by calling scipy.linalg.lu_factor, converting the piv
    vector using p_to_q, and solving two triangular linear systems
    using scipy.linalg.solve_triangular.
    """
    LU, P = sla.lu_factor(A)
    L, U = np.tril(LU, k=-1) + np.eye(A.shape[0]), np.triu(LU)
    q = p_to_q(P)
    pb = np.empty((P.shape[0],))
    for i in range(0, P.shape[0]):
        pb[i] = b[q[i]]

    y = sla.solve_triangular(L, pb, lower=True)

    return sla.solve_triangular(U, y, lower=False)








if __name__ == "__main__":
    import doctest
    import matplotlib.pyplot as plt
    import time
    doctest.testmod()
    # Q3_c()
    #test your solve_plu function on a random system
    n = 10
    A = np.random.uniform(-1,1,[n,n])
    b = np.random.uniform(-1,1,n)
    xtrue = sla.solve(A,b)
    x = solve_plu(A,b)
    print("solve_plu works:",np.allclose(x,xtrue,rtol=1e-10,atol=0))

    size, algorithm1, algorithm2 = Q3_c()
    plt.plot(size, algorithm1, label="Performance of algorithm 1")
    plt.plot(size, algorithm2, label="Performance of algorithm 2")
    plt.xlabel("Matrix size")
    plt.ylabel("Runtime")
    plt.title("Runtime of algorithm 1 vs algorithm 2")
    plt.legend()
    plt.savefig("3c.png")
    plt.show()