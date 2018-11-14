import matplotlib.pyplot as plt
from numpy import linspace, polyval
import pickle

x_data = []
y_data = []
fCoefficients = []
fPrimeCoefficients = []


def showGraphs():
    """
    Plots data points. Function modeling data. Derivative of function modeling data.
    """
    setData()

    fig, plots = plt.subplots(2,1,constrained_layout=True,figsize=(5,7)) #5,7 found by trial. figsize has odd behavior
    fig.suptitle('Military Time V. Number Crimes in LA.')

    f_x = linspace(0.0,2400,200)
    f_y = polyval(fCoefficients,f_x)
    plots[0].set_title('f(x) Modeling Data.')
    plots[0].set_xlabel('Military Time')
    plots[0].set_ylabel('Number Crimes Reported')
    plots[0].plot(x_data,y_data,'bo')
    plots[0].plot(f_x,f_y,'-')

    f_prime_x = linspace(0.0,2400,200)
    f_prime_y = polyval(fPrimeCoefficients,f_prime_x)
    plots[1].set_title('f\'(x) Roots')
    plots[1].set_xlabel('Military Time')
    plots[1].set_ylabel('Rate of Change')
    plots[1].axhline(y=0.0,color='blue',linestyle='-') # line y = 0
    plots[1].plot(f_prime_x,f_prime_y,'-',color='red') # computed fPrime
    plt.show()

def setData():
    """
    Read all data from binary files to be plotted.
    """
    global x_data
    global y_data
    global fCoefficients
    global fPrimeCoefficients

    with open('Coefficients.bin','rb') as file:
        fCoefficients = pickle.load(file)
    with open('FPrimeCoefficients.bin','rb') as file:
        fPrimeCoefficients = pickle.load(file)
    with open('CrimeXData.bin','rb') as file:
        x_data = pickle.load(file)
    with open('CrimeYData.bin','rb') as file:
        y_data = pickle.load(file)


if __name__ == '__main__':
    print('This module visualizes crime data.')
