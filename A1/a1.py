#CSC336 Assignment #1 starter code
import numpy as np

#Q2a
def alt_harmonic(fl=np.float16):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series.

    The floating point type fl is used in the calculations.
    """
    n = 0
    s = fl(0)
    while np.add(s, fl((np.divide((-1)**(n+2), n+1, dtype=fl))), dtype=fl) != s:
        n += 1
        s = np.add(s, fl((np.divide((-1)**(n+1), n, dtype=fl))), dtype=fl)
    return [n, s]

#Q2b
#add code here as stated in assignment handout
q2b_rel_error = (np.log(2) - alt_harmonic()[1])/ np.log(2)

#Q2c
def alt_harmonic_given_m(m, fl=np.float16):
    """
    Returns the sum of the first m terms of the alternating
    harmonic series. The sum is performed in an appropriate
    order so as to reduce rounding error.

    The floating point type fl is used in the calculations.
    """
    sum = fl(0)
    while m >= 1:
        sum = np.add(sum, np.divide((-1)**(m+1), m, dtype=fl), dtype=fl)
        m -= 1
    return sum


#Q3a
def alt_harmonic_v2(fl=np.float32):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series (using
    the formula in Q3a, where terms are paired).

    The floating point type fl is used in the calculations.
    """
    n = 0
    s = fl(0)
    while np.add(s, np.divide(1,np.multiply(2*n + 2, 2*n + 1, dtype=fl), dtype=fl), dtype=fl) != s:
        n += 1
        s = np.add(s, np.divide(1, np.multiply(2*n, 2*n - 1, dtype=fl), dtype=fl), dtype=fl)
    return [n, s]

#Q3b
#add code here as stated in assignment handout
q3b_rel_error = np.divide(np.subtract(np.log(2), alt_harmonic_v2()[1]),np.log(2))

#Q4b
def hypot(a, b):
    """
    Returns the hypotenuse, given sides a and b.
    """
    c = np.sqrt((a + b + np.sqrt(2) * np.sqrt(a) * np.sqrt(b))) * np.sqrt((a + b - np.sqrt(2) * np.sqrt(a) * np.sqrt(b)))#replace with improved algorithm
    return c

#Q4c
q4c_input = [1e+200] #see handout for what value should go here.


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    print(alt_harmonic_v2(), q3b_rel_error, np.finfo(np.float32).eps /np.log(2))