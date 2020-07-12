#CSC336 - Homework 1 - starter code

#Q1a
def int2bin(x):
    """
    convert integer x into a binary bit string

    >>> int2bin(0)
    '0'
    >>> int2bin(10)
    '1010'
    """
    current_value = x
    solution = ''
    counter = 0
    while current_value != 0:
        counter += 1
        solution = str(current_value % 2) + solution
        current_value = current_value // 2
    if counter == 0:
        return '0' + solution
    return solution

#Q1b
def frac2bin(x):
    """
    convert x into its fractional binary representation.

    precondition: 0 <= x < 1

    >>> frac2bin(0.75)
    '0.11'
    >>> frac2bin(0.625)
    '0.101'
    >>> frac2bin(0.1)
    '0.0001100110011001100110011001100110011001100110011001101'
    >>> frac2bin(0.0)
    '0.0'
    """
    current_value = x
    solution = '0.'
    counter = 0
    while current_value != 0:
        counter += 1
        a = 2 * current_value
        if a >= 1:
            solution = solution + '1'
            current_value = a - 1
        else:
            solution = solution + '0'
            current_value = a
    if counter == 0:
        solution = solution + '0'
    return solution

#Question 3
#set these to the values you have chosen as your answers to
#this question. Make sure they aren't just zero and they are
#in the interval (0,1).
x1 = 2**(-67)
x2 = 2**(-25*8)


#Question 4
import numpy as np

#list of floating point data types in numpy for reference
fls = [np.float16,np.float32,np.float64]

#fix the following function so that it is correct
import math #you can remove this import once you
            #have the correct solution
def eval_with_precision(x,y,fl=np.float64):
    """
    evaluate sin((x/y)**2 + 2**2) + 0.1, ensuring that ALL
    calculations are correctly using the
    floating point type fl

    precondition: y != 0

    >>> x = eval_with_precision(2,10,fl=fls[0])
    >>> type(x) == fls[0]
    True
    >>> x == fls[0](-0.6816)
    True
    """
    return fl(np.sin(np.square(np.divide(x, y, dtype=fl), dtype=fl) + fl(2**2), dtype=fl) + fl(1)/fl(10))

def factorial(x):
    if x == 1:
        return 1
    else:
        return x* factorial(x - 1)
if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    #you can add any additional testing you do here