#CSC336 Assignment #1 starter code for the report question
import numpy as np
import matplotlib.pyplot as plt

"""
See the examples in class this week if you
aren't sure how to start writing the code
to run the experiment and produce the plot for Q1.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.

A few things you'll likely find useful:

import matplotlib.pyplot as plt

hs = np.logspace(-15,-1,15)
plt.figure()
plt.loglog(hs,rel_errs)
plt.show() #displays the figure
plt.savefig("myplot.png")



a_cmplx_number = 1j #the j in python is the complex "i"

try to reuse code where possible ("2 or more, use a for")

"""

#example function header, you don't have to use this
def fd(f, x, h, fl=np.complex_):
    """
    Return the forward finite difference approximation
    to the derivative of f(x), using the step size(s)
    in h.

    f is a function

    x is a scalar

    h may be a scalar or a numpy array (like the output of
    np.logspace(-15,-1,15) )

    """
    return np.divide(f(np.add(x, h, dtype=fl)) - f(x), h, dtype=fl)

def cd(f, x, h, fl=np.complex_):
    """
    Return the centered finite difference approximation
    to the derivative of f(x), using the step size(s)
    in h.

    f is a function

    x is a scalar

    h may be a scalar or a numpy array (like the output of
    np.logspace(-15,-1,15) )

    """
    return np.divide(np.subtract(f(np.add(x, h, dtype=fl)), f(np.subtract(x, h, dtype=fl)), dtype=fl), (2*h), dtype=fl)

def complexStep(f, x, h, fl=np.complex_):
    """
    Return the backward finite difference approximation
    to the derivative of f(x), using the step size(s)
    in h.

    f is a function

    x is a scalar

    h may be a scalar or a numpy array (like the output of
    np.logspace(-15,-1,15) )

    """
    return np.divide(np.imag(f(np.add(x, 1j*h, dtype=fl))), h, dtype=fl)

def rounding_error(h, fl=np.complex_):
    """
    Generate the rounding error
    """
    return np.divide(np.finfo(fl).eps, h)

def relative_error(value, fl=np.complex_):
    return np.abs(np.subtract(1, np.divide(value, 4.05342789389862, dtype=fl), dtype=fl))

def truncation_error(value, h, fl=np.complex_):
    return np.divide(value*np.square(h, dtype=fl), 12, dtype=fl)


def f(x, fl=np.complex_):
    numerator = np.exp(x, dtype=fl)
    denominator = np.sqrt(np.power(np.sin(x, dtype=fl), 3, dtype=fl) + np.power(np.cos(x, dtype=fl), 3, dtype=fl), dtype=fl)
    return np.divide(numerator, denominator, dtype=fl)

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("step size")
    plt.ylabel("Relative error")
    plt.title("Relative error for different derivative estimation algorithms")

    print(np.finfo(np.complex_).eps)
    h = np.arange(1e-15, 5e-1, 1e-8, dtype=np.complex_)

    r_error = np.divide(rounding_error(h), 4.05342789389862)
    t_error = np.divide(truncation_error(32.16, h), 4.05342789389862)
    # 32.16 comes from the third derivative of f at point x = 1.5

    fd = relative_error(fd(f, 1.5, h))
    csd = relative_error(complexStep(f, 1.5, h))
    cd = relative_error(cd(f, 1.5, h))

    plt.plot(h, fd, label='Forward Difference')
    fd = None

    plt.plot(h, cd, label='Centered Difference')
    cd = None

    plt.plot(h, csd, label='Complex step')
    bd = None

    plt.plot(h, t_error, label='Lower bound for theoretical Truncation Error', linestyle="--")
    t_error = None

    plt.plot(h, r_error, label='Lower bound for theoretical Rounding Error', linestyle="--")
    r_error = None

    plt.legend()
    plt.savefig("1.png")
