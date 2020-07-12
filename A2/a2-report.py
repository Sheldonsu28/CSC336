#CSC336 Assignment #2 starter code for the report question

#These are some basic imports you will probably need,
#but feel free to add others here if you need them.
import numpy as np
from scipy.sparse import diags
import scipy.linalg as sla
from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.linalg import solve, solve_triangular, solve_banded
import scipy.sparse as sp
import time
import matplotlib
import matplotlib.pyplot as plt

"""
See the examples in class this week or ask on Piazza if you
aren't sure how to start writing the code
for the report questions.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.
"""

"""
timing code sample (feel free to use timeit if you find it easier)
#you might want to run this in a loop and record
#the average or median time taken to get a more reliable timing
start_time = time.perf_counter()
#your code here to time
time_taken = time.perf_counter() - start_time
"""

def get_true_sol(n):
    """
    returns the true solution of the continuous model on the mesh,
    x_i = i / n , i=1,2,...,n.
    """
    x = np.linspace(1/n,1,n)
    d = (1/24)*(-(1-x)**4 + 4*(1-x) -3)
    return d



def compare_to_true(d):
    """
    produces plot similar to the handout,
    the input is the solution to the n x n banded linear system,
    this is one way to visually check if your code is correct.
    """
    dtrue = get_true_sol(100) #use large enough n to make plot look smooth

    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 14})
    plt.title("Horizontal Cantilevered Bar")
    plt.xlabel("x")
    plt.ylabel("d(x)")

    xtrue = np.linspace(1/100,1,100)
    plt.plot(xtrue,dtrue,'k')

    n = len(d)
    x = np.linspace(0,1,n+1)
    plt.plot(x,np.hstack([0,d]),'--r')

    plt.legend(['exact',str(n)])
    plt.grid()
    plt.show()

def initalize_matrix(n):
    """
    Initialize the matrix for the physical system
    :param n:   size of the matrix
    :return:    physical matrix
    """
    diagnal = [[6 for i in range(0, n-3)], [-4 for j in range(0, n-2)], [-4 for j in range(0, n-2)],
               [1 for j in range(0, n-2)], [1 for j in range(0, n-2)]]
    diagnal[0].insert(0, 9)
    diagnal[0].append(5)
    diagnal[0].append(1)
    diagnal[1].append(-2)
    diagnal[2].append(-2)
    matrix = diags(diagnal, [0, -1, 1, -2, 2], format='csr')
    factored_d = [[1 for r in range(0, n-1)], [-2 for r in range(0, n - 1)], [1 for r in range(0, n-2)]]
    factored_d[0].insert(0, 2)
    factored = diags(factored_d, [0, 1, 2], format='csr')
    return matrix, diagnal, factored, factored_d

# Timing code for entire Q1
def timing():
    algorithm_runtime = [[],
                         [],
                         [],
                         [],
                         [],
                         [],
                         []]
    matrix_size = []

    for m in range(1, 21):
        n = m * 100
        matrix_size.append(n)
        R, diagnal, factored, factored_d = initalize_matrix(n)
        b = -1/(n**4) * np.ones((n,))

        dense_R = R.toarray()
        dense_f = factored.toarray()
        s0 = time.perf_counter()
        r0 = sla.solve(dense_R, b)
        e0 = time.perf_counter()
        algorithm_runtime[0].append(e0 - s0)

        ab = np.ones((5, n))
        ab[2, :] = diagnal[0]
        ab[1, 1:] = diagnal[1]
        ab[3, 0:-1] = diagnal[2]
        ab[0, 2:] = diagnal[3]
        ab[4, 0:-2] = diagnal[4]
        s1 = time.perf_counter()
        r1 = sla.solve_banded((2, 2), ab, b)
        e1 = time.perf_counter()
        algorithm_runtime[1].append(e1 - s1)


        s2 = time.perf_counter()
        r2 = spsolve(R, b)
        e2 = time.perf_counter()
        algorithm_runtime[2].append(e2 - s2)

        transpose = np.transpose(dense_f)
        s3 = time.perf_counter()
        r3 = solve_triangular(dense_f, b, lower=False)
        r3 = solve_triangular(transpose, r3, lower=True)
        e3 = time.perf_counter()
        algorithm_runtime[3].append(e3 - s3)

        AB = np.ones((3, n))
        AB[2,:] = factored_d[0]
        AB[1, 1:] = factored_d[1]
        AB[0, 2:] = factored_d[2]
        AB_t = np.empty((3, n))
        AB_t[0, :] = factored_d[0]
        AB_t[1, :-1] = factored_d[1]
        AB_t[2, :-2] = factored_d[2]
        s4 = time.perf_counter()
        r4 = solve_banded((0, 2), AB, b)
        r4 = solve_banded((2, 0), AB_t, r4)
        e4 = time.perf_counter()
        algorithm_runtime[4].append(e4 - s4)

        factored_transpose = sp.csr_matrix(factored.transpose())
        s5 = time.perf_counter()
        r5 = spsolve_triangular(factored, b, lower=False)
        r5 = spsolve_triangular(factored_transpose, r5, lower=True)
        e5 = time.perf_counter()
        algorithm_runtime[5].append(e5 - s5)

        s6 = time.perf_counter()
        c, lower = sla.cho_factor(dense_R)
        r6 = sla.cho_solve((c, lower), b)
        e6 = time.perf_counter()
        algorithm_runtime[6].append(e6 - s6)


    plt.plot(matrix_size, algorithm_runtime[0], label='LU')
    plt.plot(matrix_size, algorithm_runtime[1], label='Banded')
    plt.plot(matrix_size, algorithm_runtime[2], label='Sparse LU')
    plt.plot(matrix_size, algorithm_runtime[3], label='R')
    plt.plot(matrix_size, algorithm_runtime[4], label='Banded R')
    plt.plot(matrix_size, algorithm_runtime[5], label='Sparse R')
    plt.plot(matrix_size, algorithm_runtime[6], label='Chol')
    plt.xlabel("Matrix size")
    plt.ylabel("Runtime")
    plt.title("Performance of different solver")
    plt.legend()
    plt.savefig("Q1.png")
    plt.show()
        
# Code for Q2
def get_error():
    matrix_size = []
    algorithm_error = [[],[]]
    for i in range(4, 17):
        n = 2**i
        matrix_size.append(n)
        R, diagnal, factored, factored_d = initalize_matrix(n)
        b = -1 / (n ** 4) * np.ones((n,))

        true_solution = get_true_sol(n)
        solution_norm = np.linalg.norm(true_solution, ord=np.inf)

        ab = np.ones((5, n))
        ab[2, :] = diagnal[0]
        ab[1, 1:] = diagnal[1]
        ab[3, 0:-1] = diagnal[2]
        ab[0, 2:] = diagnal[3]
        ab[4, 0:-2] = diagnal[4]
        LU_banded_result = sla.solve_banded((2, 2), ab, b)



        AB = np.ones((3, n))
        AB[2, :] = factored_d[0]
        AB[1, 1:] = factored_d[1]
        AB[0, 2:] = factored_d[2]
        AB_t = np.empty((3, n))
        AB_t[0, :] = factored_d[0]
        AB_t[1, :-1] = factored_d[1]
        AB_t[2, :-2] = factored_d[2]
        r4 = solve_banded((0, 2), AB, b)
        prefactored_banded = solve_banded((2, 0), AB_t, r4)

        prefactored_banded_norm = np.linalg.norm(prefactored_banded, ord=np.inf)
        LU_banded_norm = np.linalg.norm(LU_banded_result, ord=np.inf)

        algorithm_error[0].append(np.abs(solution_norm - LU_banded_norm)/solution_norm)
        algorithm_error[1].append(np.abs(solution_norm - prefactored_banded_norm)/solution_norm)

    plt.loglog(matrix_size, algorithm_error[0], label="Relative error for LU banded solution")
    plt.loglog(matrix_size, algorithm_error[1], label="Relative error for Prefactored R solution")
    plt.title("Relative error for LU banded solution and Prefactored R solution vs matrix size")
    plt.xlabel("Matrix size")
    plt.ylabel("Relative error")
    plt.legend()
    plt.savefig("Q2.png")
    plt.show()


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    d = np.zeros(8)
    compare_to_true(d)

    # timing()
    # get_error()
    print(np.finfo(np.float64))

