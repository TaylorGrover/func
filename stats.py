import matplotlib.pyplot as plt
import numpy as np
import os

### Set of useful statistics functions
def pearson(x,y):
    return covariance(x,y)/(stddev(x)*stddev(y))

def covariance(x,y):
    return sum((x-np.mean(x))*(y-np.mean(y)))/len(x)

def variance(x):
    return np.sqrt((sum(x**2)-sum(x)**2/len(x))/(len(x)-1))

def residuals(f,vals):
    total = 0
    avg = np.mean(vals)
    for i, val in enumerate(vals):
        total += (avg-f(val))**2
    return total

def squares(vals):
    total = 0
    avg = np.mean(vals)
    for i, val in enumerate(vals):
        total += (val-avg)**2
    return total

## Not currently a very useful function, as it uses the python input function to get values
# and would be better implemented with a gui
def getDataPoints():
    inputErrorMsg = "Invalid input"
    arr = []
    while True:
        try:
            numberOfMeasurements = int(input("Enter the number of measurements to be taken: "))
        except ValueError:
            print(inputErrorMsg)
        else:
            break
    for i in range(numberOfMeasurements):
        while True:
            try:
                arr.append(float(input("Data entry " + str(i+1) + ": ")))
            except ValueError:
                print(inputErrorMsg)
            else:
                break
    return arr

## This function already exists via np.mean
def average(vals = None):
    if vals == None:
        vals = getDataPoints()
    else:
        arr = vals
    return sum(vals)/len(vals)

# Returns the standard deviation of a set of data
def stddev(vals = None):
    #if vals == None:
    #    vals = getDataPoints()
    average_x = np.mean(vals)
    print("Average:\t" + str(average_x))
    cumulative_total = 0
    for i, x in enumerate(vals):
        cumulative_total += (average_x - x)**2
    return np.sqrt(cumulative_total/(len(vals)-1))

## This function is used to find the least squared residuals of a data set and 
# returns the slope and y-intercept of the best-fit line. Because there was not 
# an explicit method for solving for the least squared residuals that I could 
# find on Wikipedia, I had to use the equation f(m,b) = ∑[(y - mx - b)²], which
# I then set each partial derivative to zero and solved for the respective m and b
# values to yield the minimum value of f(m,b).
'''
    Equation for finding the linear regression using the least squares method:
    m=(∑y∑x−n∑yx)/((∑x)²−n∑x²)
    b=(∑y∑x²−∑x∑yx)/(n∑x²−(∑x)²)
'''
def minr(x,y):
    n = len(x)
    m = (sum(y)*sum(x)-n*sum(x*y))/(sum(x)**2 - n*sum(x**2))
    b = (sum(y)*sum(x**2) - sum(x)*sum(x*y))/(n*sum(x**2)-sum(x)**2)
    return (m,b)

def RMSE(x,y):    # Method for finding the RMSE (Root-Mean-Square error)
    m,b = minr(x,y)
    return np.sqrt(np.sum((y-m*x-b)**2))

## This function was used to test the least square residuals approach to linear regression by creating a set of random values along a domain
# and then plotting them using matplotlib.pyplot.
def testData(total=100,max_residual=10):
    fs=10
    x = np.arange(0,total,.001)
    domain=np.arange(1,total+1,1)
    fRange=[]
    for i in range(total):
        fRange.append(domain[i]+(2*max_residual*np.random.random()-max_residual+1000))
    fRange = np.array(fRange)
    plt.scatter(domain,fRange) 
    
    # use least squares method to find best fit line for data
    m,b = minr(domain,fRange)
    plt.plot(x,m*x+b)
    print(m)
    print(b)
    plt.show()
