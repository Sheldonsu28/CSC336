# CSC 336 HW#3 starter code

import scipy.linalg as sla
import numpy as np
import numpy.linalg as LA

# Q1 assign values to the following variables as
# specified in the handout

A = np.array([[21.0, 67.0, 88.0, 73.0],
                     [76.0, 63.0, 7.0, 20.0],
                     [0.0, 85.0, 56.0, 54.0],
                     [19.3, 43.0, 30.2, 29.4]])
B = np.array([[1, 2, 3, 4],
              [12, 13, 14, 5],
              [11, 16, 15, 6],
              [10, 9, 8, 7]])
C = np.transpose(B)
b = np.array([[141.0],
              [109.0],
              [218.0],
              [93.7]])
x = sla.solve(A, b)
r = np.matmul(A, x) - b
Ainv = LA.inv(A)
c1 = LA.cond(A, p=1)
c1_2 = LA.norm(A, ord=1) * LA.norm(Ainv, ord=1)
cinf = LA.cond(A, p=np.inf)
A32 = A.astype(np.float32)
b32 = b.astype(np.float32)
x32 = sla.solve(A32, b32)
y = np.matmul(np.matmul(sla.inv(B), (2*A + np.identity(A.shape[0]))), np.matmul(LA.inv(C) + A, b))


# Q2 Hilbert matrix question
# your code here
def Hilbert_matrix_error(fl=np.float64):
    print(" Results for np.float64")
    print("n | rel error |  cond(H)")
    print("-------------------------")
    n = 2
    while True:
        H = sla.hilbert(n)
        condH = LA.cond(H, p=np.inf)
        x = np.ones((n,), dtype=fl)
        b = np.matmul(H, x, dtype=fl)
        result = sla.solve(H, b)
        r_error = np.divide(sla.norm(x - result, ord=np.inf), sla.norm(x, ord=np.inf), dtype=np.float64)
        print(str(n), "|", np.format_float_scientific(r_error, precision=3), "|", np.format_float_scientific(condH, precision=3))
        n += 1
        if r_error >= 1:
            break


# Q3c
# provided code for gaussian elimination (implements algorithm from the GE notes)
def ge(A, b):
    for k in range(A.shape[0] - 1):
        for i in range(k + 1, A.shape[1]):
            if A[k, k] != 0:
                A[i, k] /= A[k, k]
            else:
                return False
            A[i, k + 1:] -= A[i, k] * A[k, k + 1:]
            b[i] = b[i] - A[i, k] * b[k]
    return True


def bs(A, b):
    x = np.zeros(b.shape)
    x[:] = b[:]
    for i in range(A.shape[0] - 1, -1, -1):
        for j in range(i + 1, A.shape[0]):
            x[i] -= A[i, j] * x[j]
        if A[i, i] != 0:
            x[i] /= A[i, i]
        else:
            return None
    return x


def ge_solve(A, b):
    if ge(A, b):
        return bs(A, b)
    else:
        return None  # GE failed


def solve(eps):
    """
    return the solution of [ eps 1 ] [x1]   [1 + eps]
                           |       | |  | = |       |
                           [ 1   1 ] [x2]   [   2   ]
    The solution is obtained using GE without pivoting
    and back substitution. (the provided ge_solve above)
    """
    A = np.array([[eps, 1],
                  [1, 1]])
    b = np.array([[np.add(1, eps, dtype=np.float64)],
                  [2]])
    x = ge_solve(A, b)  # code here
    return x

# Q3d code here for generating your table of values
def test_heath_a(fl = np.float64):
    print("K | relative error")
    print("------------------")
    for k in range(1, 11):
        eps = np.power(10, -2*k, dtype=fl)
        x = np.array([[1],
                      [1]])
        x_hat = solve(eps)
        r_error = LA.norm(x_hat - x, ord=np.inf)/LA.norm(x, ord=np.inf)
        print(str(k), "|", np.format_float_scientific(r_error, precision=3))

# if __name__ == '__main__':
#     # test_heath_a()
# #     Hilbert_matrix_error()
# #     print(np.finfo(np.float64).eps)
#     test_heath_a()