#CSC336 Summer 2020 HW5 starter code

import numpy as np

#question 1
def estimate_convergence_rate(xs):
    """
    Return approximate values of the convergence rate r and
    constant C (see section 5.4 in Heath for definition and
    this week's worksheet for the approach), for the given
    sequence, xs, of x values.

    (You might find np.diff convenient)

    Examples:

    >>> xs = 1+np.array([1,0.5,0.25,0.125,0.0625,0.03125])
    >>> b,c = estimate_convergence_rate(xs)
    >>> close = lambda x,y : np.allclose(x,y,atol=0.1)
    >>> close(b,1)
    True
    >>> close(c,0.5)
    True
    >>> xs = [0, 1.0, 0.7357588823428847, 0.6940422999189153,\
    0.6931475810597714, 0.6931471805600254, 0.6931471805599453]
    >>> b,c = estimate_convergence_rate(xs)
    >>> close(b,2)
    True
    >>> close(c,0.5)
    True
    """
    diff = np.diff(xs)
    r = np.log(abs(diff[-1]/diff[-2]))/np.log(abs(diff[-2]/diff[-3]))
    c = np.abs(xs[-1] - xs[-2])/(np.abs(xs[-2] - xs[-3])**r)

    return r, c #r,c

#question 2
#Put any functions you define for Q2 here.
#See the worksheet for a similar example if you have
#trouble getting started with Q2.
def g1(x, derivative=0):
    if derivative == 0:
        return (np.square(x) + 2)/3
    return 2*x/3

def g2(x, derivative=0):
    if derivative == 0:
        return np.sqrt(3*x - 2)
    return 3/(2*np.sqrt(3*x - 2))

def g3(x, derivative=0):
    if derivative == 0:
        return 3 - 2/x
    return 2/np.square(x)

def g4(x, derivative=0):
    if derivative == 0:
        return (np.square(x) - 2)/(2*x - 3)
    return 2*(np.square(x) - 3*x + 2)/np.square(2*x - 3)

def Q2b():
    x = np.arange(0.5, 4, 0.01)

    import matplotlib.pyplot as plt

    plt.plot(x, g1(x, 1), label='Derivative of g1(x)')

    plt.plot(x, g2(x, 1), label='Derivative of g2(x)')

    plt.plot(x, g3(x, 1), label='Derivative of g3(x)')

    plt.plot(x, g4(x, 1), label='Derivative of g4(x)')

    plt.ylim([-5, 6])
    plt.title('Derivative of different function')
    plt.legend()
    plt.savefig('Q2b.png')

def Q2c(guess, function, title):
    import matplotlib.pyplot as plt
    iteration = 0
    x = guess
    xs = []
    ys = []
    while iteration < 1000:
        y = function(x)
        xs.append(x)
        ys.append(y)
        iteration += 1
        if x == y:
            break
        x = y
    plt.plot(xs, ys, 'o')
    plt.title(title)
    plt.xlabel('x value')
    plt.ylabel('y value')
    plt.legend()
    plt.savefig('Q2c'+title+'.png')
    return xs



#optional question 3 starter code
a = np.pi #value of root
def f(x,m=2):
    return (x - a)**m
def df(x,m=2):
    return m*(x-a)**(m-1)

def newton(f,fp,xo,atol = 1e-10):
    """
    Simple implementation of Newton's method for scalar problems,
    with stopping criterion based on absolute difference between
    values of x on consecutive iterations.

    Returns the sequence of x values.
    """
    x = xo
    xs = [xo]
    err = 1
    while(err > atol):
        x = x - f(x)/fp(x)
        xs.append(x)
        err = np.abs(xs[-2]-xs[-1])
    return xs

def multiplicity_newton(f,fp,xo,atol = 1e-10):
    """
    version of Newton's method that monitors the convergence rate,
    r and/or C, and tries to speedup convergence for multiple roots

    Returns the sequence of x values.
    """
    #modify the Newton's method code below based on your algorithm
    #from Q3b
    x = xo
    xs = [xo]
    err = 1
    while(err > atol):
        x = x - f(x)/fp(x)
        xs.append(x)
        err = np.abs(xs[-2]-xs[-1])
    return xs

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #call any code related to Q2 here
    # Q2b()


    g1x = Q2c(3, g1, 'g1 with inital guess 3')
    # print(estimate_convergence_rate(g1x[0:-1]))
    # g1x = Q2c(0, g1, 'g1 with inital guess 1')
    # print(estimate_convergence_rate(g2x[0:-1]))
    # g2x1 = Q2c(4, g2, 'g2 with inital guess 4')
    # g2x1 = Q2c(1.5, g2, 'g2 with inital guess 1.5')
    # print(estimate_convergence_rate(g2x1[:-3]))
    # g3x1 = Q2c(1.5, g3, 'g3 with inital guess 1.5')
    # g3x1 = Q2c(4, g3, 'g3 with inital guess 4')
    # print(estimate_convergence_rate(g3x1))
    # g4x1 = Q2c(1.75, g4, 'g4 with inital guess 1.75')
    # g4x1 = Q2c(4, g4, 'g4 with inital guess 4')
    # print(estimate_convergence_rate(g4x1))
    #optional code if you try Q3c
    if False:
        xo = 0
        print("error orig, error alt, iters orig, iters alt")
        for m in range(2,8):
            xs = multiplicity_newton(lambda x: f(x,m=m),lambda x: df(x,m=m),xo)
            iters_alt = len(xs)
            xalt = xs[-1]
            xs = newton(lambda x: f(x,m=m),lambda x: df(x,m=m),xo)
            iters_orig = len(xs)
            x = xs[-1]
            print(f"{x-a:<10.2e},{xalt-a:<10.2e},{iters_orig:<11d},"
                    f"{iters_alt:<10d}")