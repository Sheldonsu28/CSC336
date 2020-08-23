#CSC336 Summer 2020 HW6 starter code

#some basic imports
import numpy as np
import numpy.linalg as LA
from scipy.linalg import solve
import matplotlib.pyplot as plt

#Question 1 [autotested]

#Complete the following 5 functions that compute the Jacobians
#of the 5 nonlinear systems in Heath Exercise 5.9

#NOTE: For this question, assume x.shape == (2,),
#      so you can access components of
#      x as x[0] and x[1].

def jac_a(x):
    return np.array([[2*x[0],2 *x[1]],
                     [2*x[0], -1]])

def jac_b(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([[2*x1 + x2**3, x1 * 3 * x2**2],
                     [6*x1*x2, 3*(x1**2) - 3*x2**2] ])

def jac_c(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([[1 - 2*x2, 1 - 2*x1],
                     [2*x1 - 2, 2*x2 + 2]])

def jac_d(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([[3*x1**2, -2 * x2],
                     [1 + 2*x1*x2, x1**2]])

def jac_e(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([[2 * np.cos(x1) - 5, -np.sin(x2)],
                     [-4 * np.sin(x1), 2* np.cos(x2) - 5] ])

########################################
#Question 2

#NOTE: You may use the provided code below or write your own for Q2,
#it is up to you.

#NOTE: For this question, if you use the provided code, it assumes
#      that x.shape == (3,1),
#      so you need to access components of
#      x as x[0,0], x[1,0], and x[2,0]


#useful for checking convergence behaviour of the fixed point method
from scipy.linalg import eigvals
def spectral_radius(A):
    return np.max(np.abs(eigvals(A)))

#This is essentially the same as estimate_convergence_rate from HW5,
#but for non-scalar x values.
def conv_info(xs):
    """
    Returns approximate values of the convergence rate r,
    constant C, and array of error estimates, for the given
    sequence, xs, of x values.

    Note: xs should be an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.

    This code uses the infinity norm for the norm of the errors.
    """

    errs = []
    nxs = np.diff(np.array(xs),axis=0)

    for row in nxs:
        errs.append(LA.norm(row,np.inf))

    r = np.log(errs[-1]/errs[-2]) / np.log(errs[-2]/errs[-3])
    c = errs[-1]/(errs[-2]**r)

    return r,c,errs

#functions for doing root finding
def fixed_point(g,x0,atol = 1e-14):
    """
    Simple implementation of a fixed point iteration for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    The fixed point iteration is x_{k+1} = g(x_k).

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    x = x0
    xs = [x0]
    err = 1
    while(err > atol):
        x = g(x)
        xs.append(x)
        err = LA.norm(xs[-2]-xs[-1],np.inf)
    return np.array(xs)

def newton(f,J,x0,atol = 1e-14):
    """
    Simple implementation of Newton's method for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    f is the function and J computes the Jacobian.

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    x = x0
    xs = [x0]
    err = 1
    while(err > atol):
        x = x - solve(J(x), f(x))
        xs.append(x)
        err = LA.norm(xs[-2]-xs[-1],np.inf)
    return np.array(xs)

def broyden(f,x0,atol = 1e-14):
    """
    Simple implementation of Broyden's method for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    f is the function to find the root for.

    Initially, the approximate Jacobian is set to the identity matrix.

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    B = np.identity(len(x0))
    x = x0
    xs = [x0]
    err = 1
    fx = f(x)
    while(err > atol):
        s = solve(B,-fx)
        x = x + s
        xs.append(x)
        next_fx = f(x)
        y = next_fx - fx
        #update approximate Jacobian
        B = B + (( y - B @ s) @ s.T) / (s.T @ s)
        fx = next_fx
        err = LA.norm(xs[-2]-xs[-1],np.inf)
    return np.array(xs)

#the function g from the question description,
#(i.e. x = g(x) is the fixed-point iteration)
def g(x):
    cos, sin = np.cos(x), np.sin(x)
    return np.array([(-cos[0]/81) + ((x[1]**2)/9) + sin[2]/3,
                     (sin[0]/3) + (cos[2]/3),
                     (-cos[0]/9) + (x[1]/3) + (sin[2]/6)])

#computes the jacobian of g(x)
def getG(x):
    cos,sin = np.cos(x),np.sin(x)
    return np.array([[sin[0]/81, 2*x[1]/9, cos[2]/3],
                     [cos[0]/3, 0, -sin[2]/3],
                     [sin[0]/9, 1/3, cos[2]/6]])

#x = g(x) rewritten in the form f(x), pass this into broyden / newton code
def f(x):
    return x - g(x) #pass #your code here

#computes the jacobian of f(x)
def jac(x):
    sin, cos = np.sin(x), np.cos(x)
    return np.array([[1 - sin[0] / 81, -2 * x[1] / 9, -cos[2] / 3],
                     [-cos[0] / 3, 1, sin[2] / 3],
                     [-sin[0] / 9, -1 / 3,  1 - cos[2] / 6]])

#useful to verify your jacobian is correct
def check_jac(my_f,my_J,x,verbose=True):
    """
    Returns True if my_J(x) is close to the true jacobian of my_f(x)
    """
    J = my_J(x)
    J_CS = np.zeros([len(x),len(x)])
    for i in range(len(x)):
        y = np.array(x,np.complex)
        y[i] = y[i] + 1e-10j
        J_CS[:,i] = my_f(y)[:,0].imag / 1e-10
    if not np.allclose(J,J_CS):
        if verbose:
            print("Jacobian doesn't match - check your function "
                  "and your Jacobian approximation")
            print("The difference between your Jacobian"
                  " and the complex step Jacobian was\n")
            print(J-J_CS)
        return False
    return True

########################################
#Question 3

from scipy.optimize import fsolve

#define any functions you'll use for Q3 here, but call
#them in the main block

def Q3_f(x_val):
    x, y, z = x_val[0], x_val[1], x_val[2]
    return np.array([np.sin(x) + (y**2) + np.log(z) - 3,
                    3*x + (2**y) - (z**3),
                     (x**2) + (y**2) + (z**3) - 6])

def Q3_h_deflate1(x_val, solutions):
    if len(solutions) == 0:
        return Q3_f(x_val)

    solution = x_val - np.array(solutions)
    product = np.prod(np.linalg.norm(solution, axis=1))
    return Q3_f(x_val)/product

def Q3_h_deflate2(x_val, solutions):
    if len(solutions) == 0:
        return Q3_f(x_val)

    solution = x_val - np.array(solutions)
    product = np.prod(1 + 1/np.linalg.norm(solution, axis=1))
    return Q3_f(x_val)*product

def solve_q3(f):
    root = []
    iteration = 0
    while len(root) < 4:
        x0 = -1 * np.random.uniform(-2, 0, (3,))
        solution = fsolve(f, x0)
        if np.allclose(f(solution), np.zeros((3,))):
            sucess = True
            for s in root:
                if np.allclose(solution, s):
                    sucess = False
                    break
            if sucess:
                root.append(solution)
        iteration += 1
    return root, iteration

def solve_deflate(f_s):
    root = []
    iteration = 0
    fx = lambda x: f_s(x, root)
    while len(root) < 4:

        x0 = -1 * np.random.uniform(-2, 0, (3,))
        solution = fsolve(fx, x0)
        if np.allclose(Q3_f(solution), np.zeros((3,))):
            success = True
            for s in root:
                if np.allclose(solution, s):
                    success = False
                    break
            if success:
                root.append(solution)
        iteration += 1
    return root, iteration

def Q3():
    from time import perf_counter
    runtime1, iteration1 = [], []
    runtime2, iteration2 = [], []
    runtime3, iteration3 = [], []

    for i in range(0, 100):
        a1 = perf_counter()
        as1, iter1 = solve_q3(Q3_f)
        b1 = perf_counter()
        runtime1.append((b1 - a1)*1000)
        iteration1.append(iter1)

        a2 = perf_counter()
        as2, iter2 = solve_deflate(Q3_h_deflate1)
        b2 = perf_counter()
        runtime2.append((b2 - a2)*1000)
        iteration2.append(iter2)


        a3 = perf_counter()
        as3, iter3 = solve_deflate(Q3_h_deflate2)
        b3 = perf_counter()
        runtime3.append((b3 - a3)*1000)
        iteration3.append(iter3)

    mean1, SD1, mean_iter1, std_iter1 = np.mean(runtime1), np.std(runtime1), np.mean(iteration1), np.std(iteration1)
    mean2, SD2, mean_iter2, std_iter2 = np.mean(runtime2), np.std(runtime2), np.mean(iteration2), np.std(iteration2)
    mean3, SD3, mean_iter3, std_iter3 = np.mean(runtime3), np.std(runtime3), np.mean(iteration3), np.std(iteration3)

    print("-------------------------------------------------------------------------------")
    print("|   Method    | Runtime Mean | Runtime std | Sample num mean | Sample num std |")
    print("-------------------------------------------------------------------------------")
    print("|    naive    |   {0:2.3f}ms   |   {1:2.3f}ms  |        {2:2.0f}       |       {3:2.0f}       |"
          .format(mean1, SD1, mean_iter1, std_iter1))
    print("|   deflate   |   {0:2.3f}ms   |   {1:2.3f}ms  |        {2:2.0f}       |       {3:2.0f}       |"
          .format(mean2, SD2, mean_iter2, std_iter2))
    print("| alt deflate |   {0:2.3f}ms   |   {1:2.3f}ms  |        {2:2.0f}       |       {3:2.0f}       |"
          .format(mean3, SD3, mean_iter3, std_iter3))
    print("-------------------------------------------------------------------------------")


########################################

if __name__ == '__main__':
    #import doctest
    #doctest.testmod()

    #import any other non-standard modules here
    #and run any code for generating your answers here too.

    #Any code calling functions you defined for Q2:
    xs_fix = fixed_point(g, np.array([[4], [0], [4]]))
    r_fix, c_fix, err_fix = conv_info(xs_fix)
    xs_new = newton(f, jac, np.array([4, 0, 4]))
    r_new, c_new, err_new = conv_info(xs_new)
    xs_bor = broyden(f, np.array([[4], [0], [4]]))
    a = np.array([[4], [0], [4]])
    r_bor, c_bor, err_bor = conv_info(xs_bor)
    print(r_fix, c_fix)
    print(r_new, c_new)
    print(r_bor, c_bor)

    plt.loglog(np.arange(0, len(err_fix)), err_fix, '-',label="Fix point method")
    plt.loglog(np.arange(0, len(err_new)), err_new, '-',label="Newton's method")
    plt.loglog(np.arange(0, len(err_bor)), err_bor, '-',label="Broyden's method")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.xscale("linear")
    plt.savefig("Q2.png")

    # Q3()

    #here are some sample bits of code you might have if using the
    #provided code:
    if False:
        print("Jac check: ",check_jac(f,jac,np.ones([3,1])))
        xo = 10 * np.ones([3,1]) #arbitary starting point chosen
        xs = fixed_point(g,xo)
        r,c,errs = conv_info(xs)
        print(r,c,'fixed point') #make sure to update to format any output neatly
        plt.semilogy(errs,'*-b') #the "*-" makes it easier to
                                 #see convergence behaviour in plot
        plt.show() #make sure to add plot labels if you use this code

    ####################################

    #Any code calling functions you defined for Q3: