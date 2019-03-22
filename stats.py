import matplotlib.pyplot as plt
import numpy as np
import os

def cmd(arg):
    os.system(arg)
def cls():
    cmd("cls")

### Statistics
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
def average(vals = None):
    if vals == None:
        vals = getDataPoints()
    else:
        arr = vals
    return sum(vals)/len(vals)

def stddev(vals = None):
    #if vals == None:
    #    vals = getDataPoints()
    average_x = np.mean(vals)
    print("Average:\t" + str(average_x))
    cumulative_total = 0
    for i, x in enumerate(vals):
        cumulative_total += (average_x - x)**2
    return np.sqrt(cumulative_total/(len(vals)-1))
def minr(x,y):
    m = (sum(y)-len(y)*sum(y*x)/sum(x))/(sum(x)-len(x)*sum(x**2)/sum(x))
    b = (sum(y)-(sum(x)*sum(x*y))/sum(x**2))/(len(y)-(sum(x))**2/sum(x**2))
    return (m,b)
def RMSE(x,y):    # Method for finding the RMSE (Root-Mean-Square error)
    m,b = minr(x,y)
    return np.sqrt(np.sum((y-m*x-b)**2))
def testData(total=100,max_residual=10):
    fs=10
    x = np.arange(0,total,.001)
    domain=np.arange(1,total+1,1)
    fRange=[]
    for i in range(total):
        fRange.append(-domain[i]+(2*max_residual*np.random.random()-max_residual-1000))
    fRange = np.array(fRange)
    plt.scatter(domain,fRange) 
    
    # use least squares method to find best fit line for data
    m,b = minr(domain,fRange)
    plt.plot(x,m*x+b)
    print(m)
    print(b)
    plt.show()
