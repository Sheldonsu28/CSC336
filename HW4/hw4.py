# CSC 336 HW#4 starter code

import scipy.linalg as sla
import numpy as np

# Q1 - set these to their correct values
M_35 = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0.5, 1, 0],
                 [0, -2, 0, 1]])

P_36_a = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0]])
P_36_b = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]])

A_q1c = np.array([[2, 3, -6],
                  [0.5, -15/2, 11],
                  [3 / 2, 13 / 15, 7/15]])
y_q1c = np.array([[-8],
                  [11],
                  [14/30]])

x_q1c = np.array([[-1],
                  [0],
                  [1]])


# Q3
def q3(A, B, C, b):
    y = sla.solve(C, b)
    right_side = 2*np.matmul(A, y)+2*np.matmul(np.matmul(A, A), b) + y + np.matmul(A, b)
    return sla.solve(B, right_side)


if __name__ == '__main__':
   pass