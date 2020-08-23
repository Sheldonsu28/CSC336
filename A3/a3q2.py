# CSC 336 Summer 2020 A3Q2 starter code

# Note: you may use the provided code or write your own, it is your choice.

# some general imports
import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse as sps


def get_dn(fn):
    """
    Returns dn, given a value for fn, where Ad = b + f (see handout)

    The provided implementation uses the solve_banded approach from A2,
    but feel free to modify it if you want to try to more efficiently obtain
    dn.

    Note, this code uses a global variable for n.
    """

    # the matrix A in banded format
    diagonals = [np.hstack([0, 0, np.ones(n - 2)]),  # zeros aren't used, so can be any value.
                 np.hstack([0, -4 * np.ones(n - 2), -2]),
                 np.hstack([9, 6 * np.ones(n - 3), [5, 1]]),
                 np.hstack([-4 * np.ones(n - 2), -2, 0]),  # make sure this -2 is in correct spot
                 np.hstack([np.ones(n - 2), 0, 0])]  # zeros aren't used, so can be any value.
    A = np.vstack(diagonals) * n ** 3

    b = -(1 / n) * np.ones(n)

    b[-1] += fn

    sol = solve_banded((2, 2), A, b)
    dn = sol[-1]

    return dn


def my_code_solution(fn):
    f_prime = (get_dn(2) - get_dn(1))
    print(f_prime)

    return fn - get_dn(fn)/f_prime, f_prime


if __name__ == "__main__":
    # experiment code
    print("							dn/fn/runtime                                         ")
    print(
        "--------------------------------------------------------------------------------------------------------------")
    print(
        "|   size   |        Newton's method         |             fsolve             |        Brent's method         |")
    for i in range(5, 17):
        n = 2 ** i
        # your code here
        a1 = time.perf_counter()
        solution_brentq = brentq(get_dn, 0, 0.5)
        b1 = time.perf_counter()
        runtime_brentq = np.format_float_scientific((b1 - a1)*1000, precision=4)
        s_brentq = np.format_float_scientific(solution_brentq, precision=4)
        d_brentq = np.format_float_scientific(get_dn(solution_brentq), precision=4)

        a2 = time.perf_counter()
        solution_fsolve = fsolve(get_dn, 0)
        b2 = time.perf_counter()
        runtime_fsolve = np.format_float_scientific((b2 - a2) * 1000, precision=4)
        s_fsolve = np.format_float_scientific(solution_fsolve, precision=4)
        d_fsolve = np.format_float_scientific(get_dn(solution_fsolve), precision=4)

        a3 = time.perf_counter()
        solution_newton = my_code_solution(0)
        b3 = time.perf_counter()
        runtime_mycode = np.format_float_scientific((b3 - a3) * 1000, precision=4)
        s_mycode = np.format_float_scientific(solution_newton, precision=4)
        d_mycode = np.format_float_scientific(get_dn(solution_newton), precision=4)


        print(
            "| {} | {} / {} / {}ms | {} / {} / {}ms | {} / {} / {}ms |"
                .format(n,
                        d_mycode, s_mycode, runtime_mycode,
                        d_fsolve, s_fsolve, runtime_fsolve,
                        d_brentq, s_brentq, runtime_brentq))
