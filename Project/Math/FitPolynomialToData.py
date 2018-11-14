from bisect import insort
from numpy import asarray,polyfit, uint32
from math import floor
from os.path import join
import csv
import pickle


x_data = []  # Crime times
y_data = []  # Total count for crime at same index in x_data list

def readDataIntoLists():
    """
    Reads data from csv file and puts into lists of corresponding indexes.
    x_data is crime time and same index of y_data is count for crimes at that time.
    """
    dataFile = join('..','Data','Crime_Data.csv')
    with open(dataFile,'rt') as data_file:
        first = True; # to ignore headers
        data = csv.reader(data_file, delimiter=',')
        for row in data:
            if first:
                first = False
                continue
            time = row[3]
            #round times to closest hour
            time = floor(int(time)/100) * 100
            # crime has already been found at this time
            if time in x_data:
                index = x_data.index(time)
                y_data[index] = int(y_data[index]) + 1
            # have not seen a crime at this time yet
            else:
                #insert in sorted order
                insort(x_data,time)
                #keep same indexing for y_data
                index = x_data.index(time)
                y_data.insert(index,int(1))

    # save x and y data lists to be used later
    with open('CrimeXData.bin','wb') as x_data_file:
        pickle.dump(x_data, x_data_file)
    with open('CrimeYData.bin','wb') as y_data_file:
        pickle.dump(y_data, y_data_file)

def getCoefficients():
    """
    Calculate and return a list of coefficients for the polynomial fit to the crime data.
    """
    readDataIntoLists()

    #lists have to be np arrays for polyfit function
    X_data = asarray(x_data, dtype = uint32)
    Y_data = asarray(y_data, dtype = uint32)

    coefficients = polyfit(X_data,Y_data,7) # 8 is degree of polynomial
    #to have option of reusing coefficients instead of recomputing
    with open('Coefficients.bin','wb') as file:
        pickle.dump(coefficients,file)
    return coefficients

def getLastCoefficients():
    """
    Gets the last calculated coefficients instead of recomputing.
    """
    with open('Coefficients.bin','rb') as file:
        coefficients = pickle.load(file)
    return coefficients

if __name__ == '__main__':
    print('Module reads crime data file from Data folder and fits polynomial to the data.')
    print('The degree of the polynomial has been selected specifically for the crime data.')
