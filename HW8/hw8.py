# CSC 336 Summer 2020 HW8 starter code

import numpy as np
import matplotlib.pyplot as plt

########
# Q1 code
########

# the random data
ts = np.linspace(0, 10, 11)
ys = np.random.uniform(-1, 1, ts.shape)

# points to evaluate the spline at
xs = np.linspace(ts[0], ts[-1], 201)


# implement this function
def qintrp_coeffs(ts, ys, c1=0):
    """
    return the coefficents for the quadratic interpolant,
    as specified in the Week 12 worksheet. The coefficients
    should be returned as an array with 3 columns and 1 row
    for each subinterval, so the i'th row contains a_i,b_i,c_i.

    ts are the interpolation points and ys contains the
    data values at each interpolation point

    c1 is the value chosen for c1, default is 0
    """
    # your code here to solve for the coefficients
    result = []
    last_c = c1
    last_b = 0
    for i in range(len(ts) - 1):
        temp = []
        if i == 0:
            temp.append(ys[i])
            temp.append((ys[i + 1] - ys[i]) / (ts[i + 1] - ts[i]))
            temp.append(c1)
            last_b = ys[i + 1] - ys[i]
        else:
            temp.append(ys[i])
            temp.append((ys[i + 1] - ys[i]) / (ts[i + 1] - ts[i]))
            c = (1 / (ts[i] - ts[i + 1])) * (last_c * (ts[i] - ts[i - 1]) + last_b - (ys[i + 1] - ys[i]))
            temp.append(c)
            last_c = c
            last_b = (ys[i + 1] - ys[i]) / (ts[i + 1] - ts[i])
        result.append(temp)

    return np.array(result)


# provided code to evaluate the quadratic spline
def qintrp(coeffs, h, xs):
    """
    Evaluates and returns the quadratic interpolant determined by the
    coeffs array, as returned by qintrp_coeffs, at the points
    in xs.

    h is the uniform space between the knots.

    assumes that each xs is between 0 and h*len(coeffs)
    """
    y = []
    for x in xs:
        i = int(x // h)  # get which subinterval we are in
        if i == len(coeffs):  # properly handle last point
            i = i - 1
        C = coeffs[i]
        ytmp = C[-1] * (x - (i + 1) * h)
        ytmp = (x - i * h) * (ytmp + C[-2])
        ytmp += C[0]
        y.append(ytmp)
    return np.array(y)


# define any additional functions for Q1 here

def Q1a(xs, coeffs, name):
    y = qintrp(coeffs, ts[1] - ts[0], xs)
    plt.figure()
    plt.plot(xs, y, label="Interpolated result")
    plt.plot(ts, ys, 'o', label="Original data point")
    plt.legend()
    plt.savefig(name)


########################

# define any functions for Q2 and Q3 here
def Q2():
    A = np.array([[1, -2, 4, -8, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, -1, 0, 0],
                  [0, 0, 2, 0, 0, 0, -2, 0],
                  [0, 0, 2, -12, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2, 6]])
    b = np.array([[-27],
                  [-1],
                  [-1],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0]])
    coefficents = np.linalg.solve(A, b)

    plt.figure()
    plt.plot([-2, 0, 1], [-27, -1, 0], "o", label="Original data points")
    xs1 = np.linspace(-2, 0, 201)
    xs2 = np.linspace(0, 1, 101)

    plt.plot(xs1, generate_value(coefficents[:4], xs1), label="Interpolation for interval 1")
    plt.plot(xs2, generate_value(coefficents[4:], xs2), label="Interpolation for interval 2")
    plt.title("Cubic interpolation result")
    plt.legend()
    plt.savefig("Cubic interpolation.png")

    plt.figure()
    plt.title("Interpolation First derivative")
    plt.plot(xs1, generate_derivative(coefficents[:4], xs1), label="First derivative for interval 1")
    plt.plot(xs2, generate_derivative(coefficents[4:], xs2), label="First derivative for interval 2")
    plt.legend()
    plt.savefig("Cubic interpolation derivative.png")

    plt.figure()
    plt.title("Interpolation Second derivative")
    plt.plot(xs1, generate_2nd_derivative(coefficents[:4], xs1), label="Second derivative for interval 1")
    plt.plot(xs2, generate_2nd_derivative(coefficents[4:], xs2), label="Second derivative for interval 2")
    plt.legend()
    plt.savefig("Cubic interpolation 2nd derivative.png")

def generate_value(coeff, xs):
    a, b, c, d = coeff[0], coeff[1], coeff[2], coeff[3]
    return a + b * xs + c * (xs ** 2) + d * (xs ** 3)

def generate_derivative(coeff, xs):
    a, b, c, d = coeff[0], coeff[1], coeff[2], coeff[3]
    return b + 2*c*xs + 3*d*(xs**2)

def generate_2nd_derivative(coeff, xs):
    a, b, c, d = coeff[0], coeff[1], coeff[2], coeff[3]
    return 2*c + 6*d*xs

def Q3():
    A = np.array([[1, -2, 4, -8, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, -1, 0, 0],
                  [0, 0, 2, 0, 0, 0, -2, 0],
                  [0, 1, -4, 12, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 2, 12]])
    b = np.array([[-27],
                  [-1],
                  [-1],
                  [0],
                  [0],
                  [0],
                  [21],
                  [-3]])
    coefficents = np.linalg.solve(A, b)

    A = np.array([[1, -2, 4, -8, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, -1, 0, 0],
                  [0, 0, 2, 0, 0, 0, -2, 0],
                  [0, 0, 2, -12, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2, 6]])
    b = np.array([[-27],
                  [-1],
                  [-1],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0]])
    coefficents0 = np.linalg.solve(A, b)

    plt.figure()
    plt.plot([-2, 0, 1], [-27, -1, 0], "o", label="Original data points")
    xs1 = np.linspace(-2, 0, 201)
    xs2 = np.linspace(0, 1, 101)

    plt.plot(xs1, generate_value(coefficents[:4], xs1), label="Clamp interpolation for interval 1")
    plt.plot(xs2, generate_value(coefficents[4:], xs2), label="Clamp interpolation for interval 2")

    plt.plot(xs1, generate_value(coefficents0[:4], xs1), label="Interpolation for interval 1")
    plt.plot(xs2, generate_value(coefficents0[4:], xs2), label="Interpolation for interval 2")

    plt.title("Cubic clamp interpolation result")
    plt.legend()
    plt.savefig("Cubic interpolationQ3.png")

    plt.figure()
    plt.title("Clamp interpolation First derivative")
    plt.plot(xs1, generate_derivative(coefficents[:4], xs1), label="First derivative for interval 1")
    plt.plot(xs2, generate_derivative(coefficents[4:], xs2), label="First derivative for interval 2")
    plt.legend()
    plt.savefig("Cubic interpolation derivativeQ3.png")

    plt.figure()
    plt.title("Clamp interpolation Second derivative")
    plt.plot(xs1, generate_2nd_derivative(coefficents[:4], xs1), label="Second derivative for interval 1")
    plt.plot(xs2, generate_2nd_derivative(coefficents[4:], xs2), label="Second derivative for interval 2")
    plt.legend()
    plt.savefig("Clamp interpolation 2nd derivative Q3.png")

if __name__ == '__main__':
    # add any code here calling the functions you defined above
    # coeffs = qintrp_coeffs(ts, ys)
    # new_x = np.linspace(0, 10, 100)
    # Q1a(new_x, coeffs, "Q1b.png")
    # ys[0] += 0.5
    # coeffs1 = qintrp_coeffs(ts, ys)
    # Q1a(new_x, coeffs1, "Q1b_add.png")
    Q2()
    Q3()
