# CSC 336 Summer 2020 A3Q3 starter code

# Please ask if you aren't sure what any part of the code does.
# We will do a similar example on the Week 10 worksheet.

# some general imports
from time import perf_counter
import datetime
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # for plotting dates

# loads the data from the provided csv file
# the data is stored in data, which will be
# a numpy array - the first column is the number of recovered individuals
#              - the second column is the number of infected individuals
# the data starts on '2020/03/05' (March 5th) and is daily.
data = []
with open("ONdata.csv") as f:
    l = f.readline()
    # print(l.split(',')) #skip the column names from the csv
    l = f.readline()
    while '2020/03/05' not in l:
        l = f.readline()
    e = l.split(',')
    data.append([float(e[5]) - float(e[12]), float(e[12])])
    for l in f:
        e = l.split(',')
        data.append([float(e[5]) - float(e[12]), float(e[12])])
data = np.array(data)

# the 3 main parameters for the model, we'll use them as
# global variables, so you can refer to them anywhere
beta0 = 0.32545622
gamma = 0.09828734
w = 0.75895019

# simulation basic scenario (see the simulation function later)
base_ends = [28, 35, 49, 70, 94, 132]
base_beta_factors = [1, 0.57939203, 0.46448341,
                     0.23328388, 0.30647815,
                     0.19737586]

# we'll also use beta as a global variable
beta = beta0

# We are assuming no birth or death rate, so N is a constant
N = 15e6

# assumed initial conditions
E = 0  # assumed to be zero
I = 18  # initial infected, based on data
S = N - E - I  # everyone else is susceptible
R = 0
x0 = np.array([S, E, I, R])


# The right hand side function for the SEIR model ODE, x'(t) = F(x(t))
# Feel free to use this when implementing the 3 methods, but you don't have to.
def F(x):
    return np.array([-x[0] * (beta / N * x[2]),
                     (beta / N * x[0] * x[2] - w * x[1]),
                     (w * x[1] - gamma * x[2]),
                     x[2] * (gamma)
                     ])


# your three numerical methods for performing a single time step
def method_I(x, h):
    return np.array([x[0] - h * (beta / N) * x[0] * x[2],
                     x[1] + h * (beta / N) * x[0] * x[2] - w * h * x[1],
                     x[2] + w * h * x[1] - h * gamma * x[2],
                     x[3] + h * gamma * x[2]])


def method_II(x, h):
    f1 = lambda x0: np.array([x0[0] - x[0] + h * (beta / N) * x0[0] * x0[2],
                              x0[1] - x[1] - h * (beta / N) * x0[0] * x0[2] + w * h * x0[1],
                              x0[2] - x[2] - w * h * x0[1] + h * gamma * x0[2],
                              x0[3] - x[3] - h * gamma * x0[2]
                              ])

    jacob = lambda x1: np.array([[1 + h * (beta / N) * x1[2], 0, h * (beta / N) * x1[0], 0],
                                 [-h * (beta / N) * x1[2], 1 + w * h, -h * (beta / N) * x1[0], 0],
                                 [0, -w * h, 1 + h * gamma, 0],
                                 [0, 0, -h * gamma, 1]])
    solution = fsolve(f1, x, fprime=jacob)
    return solution


def method_III(x, h):
    f1 = lambda x0: np.array([x0[0] - x[0] + 0.5 * h * (beta / N) * (x0[0] * x0[2] + x[0] * x[2]),
                              x0[1] - x[1] - 0.5 * h * (beta / N) * (x0[0] * x0[2] + x[0] * x[2]) + w * 0.5 * h * (
                                      x0[1] + x[1]),
                              x0[2] - x[2] - w * 0.5 * h * (x0[1] + x[1]) + 0.5 * h * gamma * (x0[2] + x[2]),
                              x0[3] - x[3] - 0.5 * h * gamma * (x0[2] + x[2])])

    jacob = lambda x1: np.array([[1 + h * (beta / (2 * N)) * x1[2], 0, h * (beta / (2 * N)) * x1[0], 0],
                                 [-h * (beta / (2 * N)) * x1[2], 1 + w * (h / 2), -h * (beta / (2 * N)) * x1[0], 0],
                                 [0, -w * (h / 2), 1 + (h / 2) * gamma, 0],
                                 [0, 0, -(h / 2) * gamma, 1]])

    solution = fsolve(f1, x, fprime=jacob)

    return solution


METHODS = {'I': method_I, 'II': method_II, 'III': method_III}


# take a step of length 1 using n smaller steps
# used in simulation (see below)
def step(x, n, method):
    # simulate step_length time unit using n small uniform length steps
    # and return the new state
    for i in range(n):
        x = method(x, 1 / n)
    return x


def ode_solver(x, start, end):
    from scipy.integrate import solve_ivp as ode
    fun = lambda t, x: F(x)
    sol = ode(fun, [start, end], x, t_eval=range(start, end + 1),
              method='LSODA', rtol=1e-8, atol=1e-5)
    solution = []
    for y in sol.y.T[1:, :]:
        solution.append(y.T)
    return solution


# The main simulation code:
# The simulation starts at time 0 and goes up to time ends[-1],
# the state x(t) is returned at each time 0,1,...,ends[-1].
# Inputs:
# x = initial conditions
# n = integer number of steps of method used to advance one time step
# method = 1,2, or 3 to specify which method to use. If None,
#          the builtin ODE solver is used.
# ends = list of times to break the simulate
# beta_factors = list of factors to multiply beta0 by to
#                obtain beta. E.g. on the first segment of the simulation
#                from time t=0 up to ends[0], beta = beta0 * beta_factors[0].
def simulation(x=x0, n=1, method=None,
               ends=base_ends,
               beta_factors=base_beta_factors):
    cur_time = 0
    xs = [x]
    for i, end in enumerate(ends):
        global beta
        beta = beta0 * beta_factors[i]

        if method == None:
            xs.extend(ode_solver(xs[-1], cur_time, end))
            cur_time = end
        else:
            while cur_time < end:
                xs.append(step(xs[-1], n, METHODS[method]))
                cur_time += 1
    return np.array(xs)


# some helper code to plot simulation trajectories,
# feel free to modify as needed.
def plot_trajectories(xs=data, sty='--k', label="data"):
    start_date = datetime.datetime.strptime("2020-03-05", "%Y-%m-%d")
    dates = [start_date]
    while len(dates) < len(xs):
        dates.append(dates[-1] + datetime.timedelta(1))

    # code to get matplotlib to display dates on the x-axis
    ax = plt.gca()
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)

    plt.plot(dates, xs, sty, linewidth=1, label=label)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # plt.show()


# example code to plot trajectories of I and R for each method
def plot_all_methods():
    plot_trajectories()  # plots data
    xs = simulation()

    plot_trajectories(xs[:, 2:], '-.k', 'ode')

    xs = simulation(n=10, method="II")
    plot_trajectories(xs[:, 2:], '--r', 'II')

    xs = simulation(n=10, method='I')
    plot_trajectories(xs[:, 2:], '--b', 'I')

    xs = simulation(n=10, method="III")
    plot_trajectories(xs[:, 2:], '--g', 'III')

    plt.legend()
    plt.show()


# code for plot from the handout
def make_handout_data_plot():
    plt.rcParams.update({'font.size': 16})
    plt.figure()

    plot_trajectories(xs=data[:, 0], sty="-b")
    plot_trajectories(xs=data[:, 1], sty='-r')

    plt.grid(True, which='both')
    plt.title("Ontario Covid-19 Data")
    plt.legend(["Recovered", "Infected"])
    plt.ylabel("# of people in state")
    plt.tight_layout()
    plt.show()

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(10000))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(2500))
    plt.ylim([0, 37500])


#####
# Part (c)
# add your code for your experiment to check
# accuracy and efficiency of the three methods
# Note: you should be calling simulation and varying n and method
#####

err_M1 = []
err_M2 = []
err_M3 = []
runtime1 = []
runtime2 = []
runtime3 = []
step_size = []
for i in range(1, 51):
    step_size.append(1/i)

    a1 = perf_counter()
    M1 = simulation(n=i, method='I', ends=[base_ends[-1]])
    b1 = perf_counter()
    runtime1.append((b1 - a1)*1000)

    a2 = perf_counter()
    M2 = simulation(n=i, method='II', ends=[base_ends[-1]])
    b2 = perf_counter()
    runtime2.append((b2 - a2) * 1000)

    a3 = perf_counter()
    M3 = simulation(n=i, method='III', ends=[base_ends[-1]])
    b3 = perf_counter()
    runtime3.append((b3 - a3) * 1000)

    tru_solution = simulation(n=i, ends=[base_ends[-1]])
    err_M1.append(np.linalg.norm(abs((tru_solution[-1] - M1[-1]) / tru_solution[-1])))
    err_M2.append(np.linalg.norm(abs((tru_solution[-1] - M2[-1]) / tru_solution[-1])))
    err_M3.append(np.linalg.norm(abs((tru_solution[-1] - M3[-1]) / tru_solution[-1])))

plt.title("Relative error vs step size")
plt.plot(step_size, err_M1, label="Method 1")
plt.plot(step_size, err_M2, label="Method 2")
plt.plot(step_size, err_M3, label="Method 3")
plt.xlabel("step size")
plt.ylabel("relative error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("Q3cError.png")
plt.show()
# #
# plt.title("Runtime vs step size")
# plt.plot(step_size, runtime1, label="Method 1")
# plt.plot(step_size, runtime2, label="Method 2")
# plt.plot(step_size, runtime3, label="Method 3")
# plt.xlabel("Step size")
# plt.ylabel("Runtime (ms)")
# plt.legend()
# plt.savefig("Q3cRuntime.png")
# plt.show()


#####
# Part (d)
# add your code for similar experiment, but for a fixed value of n,
# plot the error vs simulation time to see how the error changes
# over the course of the simulation
#####
# M1 = simulation(n=2, method='I')
# M2 = simulation(n=2, method='II')
# M3 = simulation(n=2, method='III')
# tru_solution = simulation(n=2)
#
# err1 = []
# err2 = []
# err3 = []
# for i in range(0, len(tru_solution)):
#     err1.append(abs(M1[i][-1] - tru_solution[i][-1]) / tru_solution[i][-1])
#     err2.append(abs(M2[i][-1] - tru_solution[i][-1]) / tru_solution[i][-1])
#     err3.append(abs(M3[i][-1] - tru_solution[i][-1]) / tru_solution[i][-1])
#
# plt.title("Relative error vs iteration")
# plt.plot(range(0, len(tru_solution)), err1, label="Method 1")
# plt.plot(range(0, len(tru_solution)), err2, label="Method 2")
# plt.plot(range(0, len(tru_solution)), err3, label="Method 3")
# plt.xlabel("iteration")
# plt.ylabel("relative error")
# plt.legend()
# plt.savefig("Q3dError.png")
# plt.show()



##########
# Part (e)
# setup your own experiment to investigate a scenario
# what exactly you do is up to you, but you will most likely
# want to pass in different values for ends and beta_factors
# for your calls to simulation
##########
def plot_simulation():
    plot_trajectories()  # plots data
    xs = simulation()

    xs = simulation(n=10, beta_factors=[1,  0.23328388,  0.23328388,
                     0.23328388, 0.23328388,
                     0.23328388])
    plot_trajectories(xs[:, 2:], '--g', 'simulation')

    plt.legend()
    plt.savefig("Q3e.png")
    plt.show()

plot_simulation()



