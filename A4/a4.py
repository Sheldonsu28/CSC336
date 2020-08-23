import numpy as np
from matplotlib import pyplot as plt
from numpy import format_float_scientific as ffsc
import scipy as sp
from scipy import interpolate
import time

years = np.linspace(1900, 1980, 9)
pops = np.array([76212168,
                 92228496,
                 106021537,
                 123202624,
                 132164569,
                 151325798,
                 179323175,
                 203302031,
                 226542199])

new_pop = np.array([76212168,
                    92228496,
                    106021537,
                    123202624,
                    132164569,
                    151325798,
                    179323175,
                    203302031,
                    226542199,
                    246709873])


def Q1():
    matrix_1 = np.vander(years)
    matrix_2 = np.vander(years - 1900)
    matrix_3 = np.vander(years - 1940)
    matrix_4 = np.vander((years - 1940) / 40)
    print("+-------------+--------------+--------------+--------------+--------------+")
    print("|    Basis    |   Basis 1    |   Basis 2    |   Basis 3    |   Basis 4    |")
    print("+-------------+--------------+--------------+--------------+--------------+")
    print("|Condition Num|  {}  |  {}  |  {}  |  {}  |".format(ffsc(np.linalg.cond(matrix_1), 4, trim='k'),
                                                               ffsc(np.linalg.cond(matrix_2), 4, trim='k'),
                                                               ffsc(np.linalg.cond(matrix_3), 4, trim='k'),
                                                               ffsc(np.linalg.cond(matrix_4), 4, trim='k')))
    print("+-------------+--------------+--------------+--------------+--------------+")


def Q2():
    matrix_4 = np.vander((years - 1940) / 40)
    plt.figure()
    plt.plot(years, pops, "o", label="Original data points")
    coeffs = np.linalg.solve(matrix_4, pops)
    xs = np.linspace(1900, 1980, 1000)
    ys = np.polyval(coeffs, (xs - 1940) / 40)
    plt.plot(xs, ys, label="Interpolation with basis 4")
    plt.legend()
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.title("Polyfit with basis 4")
    plt.savefig("Q2.png")
    plt.show()


def Q3():
    f = interpolate.pchip(years, pops)
    plt.plot(years, pops, "o", label="Original data points")
    xs = np.linspace(1900, 1980, 1000)
    ys = f(xs)
    plt.plot(xs, ys, label="Interpolation with PCHIP")
    plt.legend()
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.title("Polyfit with PCHIP")
    plt.savefig("Q3.png")
    plt.show()


def Q4():
    f = interpolate.CubicSpline(years, pops, bc_type='natural')
    plt.plot(years, pops, "o", label="Original data points")
    xs = np.linspace(1900, 1980, 1000)
    ys = f(xs)
    plt.plot(xs, ys, label="Interpolation with CubicSpline")
    plt.legend()
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.title("Polyfit with CubicSpline")
    plt.savefig("Q4.png")
    plt.show()


def Q5():
    f_cubic = interpolate.CubicSpline(years, pops, bc_type='natural')
    xs = np.linspace(1900, 1990, 1000)
    ys_cubic = f_cubic(xs)

    matrix_4 = np.vander((years - 1940) / 40)
    coeffs = np.linalg.solve(matrix_4, pops)
    ys_poly = np.polyval(coeffs, (xs - 1940) / 40)

    f_hermit = interpolate.pchip(years, pops)
    ys_hermit = f_hermit(xs)

    plt.figure()
    plt.plot(xs, ys_hermit, label="Interpolation with PCHIP")
    plt.plot(xs, ys_cubic, label="Interpolation with CubicSpline")
    print(ys_hermit[-1], ys_cubic[-1])
    plt.plot(xs, ys_poly, label="Interpolation with Poly fit")
    plt.plot(1990, 248709873, "ro", label="Real data at 1990")
    plt.title("Interpolate population up to 1990")
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.legend()
    plt.savefig("Q5.png")
    plt.show()


def Q6():
    xs = np.linspace(1900, 1980, 1000)
    f_lagrange = interpolate.lagrange((years - 1940) / 40, pops)

    a1 = time.perf_counter()
    ys_lagrange = f_lagrange((xs - 1940) / 40)
    b1 = time.perf_counter()

    f_cubic = interpolate.CubicSpline(years, pops, bc_type='natural')
    a2 = time.perf_counter()
    ys_cubic = f_cubic(xs)
    b2 = time.perf_counter()

    matrix_4 = np.vander((years - 1940) / 40)
    coeffs = np.linalg.solve(matrix_4, pops)
    a3 = time.perf_counter()
    ys_poly = np.polyval(coeffs, (xs - 1940) / 40)
    b3 = time.perf_counter()

    print("Lagrange time: {}ms".format((b1 - a1) * 1000))
    print("Cubic time: {}ms".format((b2 - a2) * 1000))
    print("Poly time: {}ms".format((b3 - a3) * 1000))
    plt.plot(xs, ys_lagrange, label="Interpolation with lagrange")
    plt.plot(years, pops, "o", label="Original data points")
    plt.plot(xs, ys_cubic, label="Interpolation with CubicSpline")
    plt.plot(xs, ys_poly, label="Interpolation with Poly fit")
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.title("Result for different interpolant")
    plt.legend()
    plt.savefig("Q6.png")
    plt.show()


def Q7():
    A1 = np.zeros((9, 9))
    A1[:, 0] = np.ones(9)
    for i in range(0, 9):
        for j in range(0, i + 1):
            array = years[i] - years
            product = np.prod(array[0:j])
            A1[i, j] = product

    new_xs = np.linspace(1900, 1990, 10)
    A2 = np.zeros((10, 10))
    A2[:, 0] = np.ones(10)
    for i in range(0, 10):
        for j in range(0, i + 1):
            array = new_xs[i] - new_xs
            product = np.prod(array[0:j])
            A2[i, j] = product

    coeff = np.linalg.solve(A1, pops)
    coeff1 = np.linalg.solve(A2, new_pop)

    xs = np.linspace(1900, 1990, 1000)
    plt.figure()
    plt.plot(new_xs, new_pop, "o", label="Original data points")
    plt.plot(xs, plot_newton_poly(years, xs, coeff), label="Newton's method degree 8")
    plt.plot(xs, plot_newton_poly(new_xs, xs, coeff1), label="Newton's method degree 9")
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.title("Newton basis interpolation for degree 8 and 9")
    plt.legend()
    plt.savefig("Q7.png")


def plot_newton_poly(x_points, x, coeff):
    n = len(x_points) - 1
    y = coeff[n]
    for i in range(1, n + 1):
        y *= (x - x_points[n - i])
        y += coeff[n - i]
    return y


def Q8():
    new_pop = np.round(pops, -6)
    xs = np.linspace(1900, 1980, 1000)
    matrix_4 = np.vander((years - 1940) / 40)

    coeffs1 = np.linalg.solve(matrix_4, pops)
    coeffs2 = np.linalg.solve(matrix_4, new_pop)
    ys = np.polyval(coeffs1, (xs - 1940) / 40)
    ys1 = np.polyval(coeffs2, (xs - 1940) / 40)

    plt.plot(years, pops, "o", label="Original data points")
    plt.plot(years, new_pop, "o", label="Rounded data points")
    plt.plot(xs, ys, "--", label="Original Interpolation")
    plt.plot(xs, ys1, "-.", label="Rounded Interpolation")
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.title("Original interpolation vs Rounded interpolation")
    plt.legend()
    plt.savefig("Q8.png")
    print(coeffs1)
    print(coeffs2)
    plt.show()


if __name__ == "__main__":
    # Q1()
    # Q2()
    # Q3()
    # Q4()
    # Q5()
    # Q6()
    # Q7()
    Q8()
