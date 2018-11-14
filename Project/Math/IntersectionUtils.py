from scipy.optimize import fsolve
from numpy import roots
import pickle

fCoefficients = []
fPrimeCoefficients = []

def f(x,coefficients):
    """
    Evaluates coefficients as polynomial function at value x
    """
    total = 0
    pow = len(coefficients) - 1
    for i in range(len(coefficients)):
        if pow == 0:
            #print('Adding: ', fCoefficients[i])
            total += coefficients[i]
        else:
            #print('Adding: ', fCoefficients[i] * (x ** pow))
            total += coefficients[i] * (x ** pow)
        pow -= 1
    return total

def calculateFPrime(f):
    """
    Find coefficients for fPrime of f assuming f is simple non-fractional polynomial function.
    """
    fPrime = []
    pow = len(f) - 1
    for i in range(len(f) -1):
        fPrimeCoefficients.append(f[i] * pow)
        fPrime.append(f[i] * pow)
        pow -= 1
    return fPrime

def y0(x):
    """
    Represents line y = 0 to for finding x intersection points.
    """
    return 0

def findIntersections(function1,function2=y0,x0=0.0):
    """
    Returns a list of all interction points of given two functions.
    Default use is passing one function and finding x intersection points.
    Returns last intersetion found.
    """
    return fsolve(lambda x : function1(x) - function2(x),x0)

def findRoots(functionCoefficients):
    """
    Returns all non imaginary roots of a function given list of function coefficients.
    """
    return roots(fPrimeCoefficients)


if __name__ == '__main__':
    print('Module provides functions for finding f\'(x) coefficients, ',end='')
    print('interction point of two functions and real root values of funtion.')
