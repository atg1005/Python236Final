import pickle
import time
from numpy import isreal
import FitPolynomialToData
import VisualizeData
import IntersectionUtils

def serilizeObj(data,fileName):
    """
    Given an object and file name / path serialize the object
    """
    with open(fileName,'wb') as file:
        pickle.dump(data,file)

def restrictToDomain(possiblePoints):
    """
    Restricts the list of roots found to the domain of the crime data.
    """
    intersectionPoints = []
    for i in range(len(possiblePoints)):
    # Must be real number and in domain of  military time
        if isreal(possiblePoints[i]) and possiblePoints[i] >= 0 and possiblePoints[i] <= 2300:
            intersectionPoints.append(possiblePoints[i])
    return intersectionPoints

if __name__ == '__main__':
    start = time.time()
    fCoefficients = FitPolynomialToData.getCoefficients() # calculate new coefficients
    #fCoefficients = FitPolynomialToData.getLastCoefficients() # load previously calculated coefficients
    fPrimeCoefficients = IntersectionUtils.calculateFPrime(fCoefficients)
    # to be used in visualization
    serilizeObj(fPrimeCoefficients,'FPrimeCoefficients.bin')

    intersectionPoints = IntersectionUtils.findRoots(fPrimeCoefficients)
    intersectionPoints = restrictToDomain(intersectionPoints)

    #evaule all peaks could be all minimums which is a potential issue.
    evaluatedValues = []
    for x in intersectionPoints:
        evaluatedValues.append(IntersectionUtils.f(x,fCoefficients))

    end = time.time()
    ex_time = end - start
    #print max found
    print('Hour:', int(round(intersectionPoints[evaluatedValues.index(max(evaluatedValues))]/100)*100),'with ',end='')
    print(int(round(max(evaluatedValues))),' crimes. Execution time:', round(ex_time,4))

    VisualizeData.showGraphs()
