import numpy as np
from scipy import interpolate as interp
from matplotlib import pyplot as plt


# Q2
def max_error(x, xs, equaldistance=True):
    if not equaldistance:
        return abs((1 / (np.math.factorial(6))) * np.prod(x - xs))
    return abs((1 / (np.math.factorial(6))) * ((xs[-1] - xs[0]) ** 6) / (2 ** 6))


def Q2():
    x0 = np.arange(0, (np.pi / 2) + (np.pi / 199), np.pi / 199)
    x = np.arange(0, (np.pi / 2) + np.pi / 8, np.pi / 8)
    y = np.sin(x)
    c = np.polyfit(x, y, 4)
    plt.plot(x, y, 'bo', label="Data point")
    plt.plot(x0, np.sin(x0), label="sin(x)")
    plt.plot(x0, np.polyval(c, x0), label="polyfit result")
    error_bound_l = []
    error_bound_u = []
    error = []

    for x1 in x0:
        error.append(max_error(x1, x))
        error_bound_l.append(np.sin(x1) - max_error(x1, x))
        error_bound_u.append(np.sin(x1) + max_error(x1, x))

    print(abs(np.polyval(c, x0)[33] - np.sin(x0)[33]), error[33], x0[33])
    print(abs(np.polyval(c, x0)[55] - np.sin(x0)[55]), error[55], x0[55])
    print(abs(np.polyval(c, x0)[64] - np.sin(x0)[64]), error[64], x0[64])

    plt.plot(x0, error_bound_l, "--", label="Lower error limit")
    plt.plot(x0, error_bound_u, "--", label="Upper error limit")
    plt.legend()
    plt.savefig("Q2.png")
    plt.show()


# Q3
def Runge(x):
    return 1 / (1 + (25 * x ** 2))


def Q3(points_num):
    xs = np.arange(-1, 1 + 2 / (points_num - 1), 2 / (points_num - 1))
    x0 = np.arange(-1, 1 + 2 / 199, 2 / 199)
    c = np.polyfit(xs, Runge(xs), points_num - 1)
    fit_result = np.polyval(c, x0)
    cubic_interpolation = interp.CubicSpline(xs, Runge(xs))

    plt.plot(xs, Runge(xs), "bo", label="Data point")
    plt.plot(x0, cubic_interpolation(x0), "--", label="CubicSpline with n = {}".format(points_num))
    plt.plot(x0, fit_result, "--", label="Polyfit with n = {}".format(points_num))
    plt.plot(x0, Runge(x0), label="Original function")
    plt.title("Interpolation with n = {} points".format(points_num))
    plt.legend()
    plt.savefig("Q3_n{}.png".format(points_num))
    plt.show()


if __name__ == "__main__":
    Q2()
    error = 1
    points = 5
    while error > 1e-10:
        points += 1
        xs = np.arange(0, (np.pi / 2), np.pi / (2 * points))
        error = max_error(np.pi / 2, xs, equaldistance=False)
    print(points)
    # Q3(21)
