#CSC336 Assignment #1 starter code for the report question
import numpy as np

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
def fd(f, x, h):
    """
    Return the forward finite difference approximation
    to the derivative of f(x), using the step size(s)
    in h.

    f is a function

    x is a scalar

    h may be a scalar or a numpy array (like the output of
    np.logspace(-15,-1,15) )

    """
    return 0 #your code here


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    pass