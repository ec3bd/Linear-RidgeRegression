#Eamon Collins
#ec3bd
#Tested using python 3.4.
#.plot() commands taken out because they're annoying, all graphs in written portion

from numpy import *
import matplotlib.pyplot as plt

def main():
    xVals,yVals = loadDataSet("Q2data.txt")

    print(standRegres(xVals, yVals))
    print(polyRegres(xVals, yVals))

def loadDataSet(filename):
    file = open(filename, 'r')
    xVals = []
    yVals = []
    for line in file:
        line = line.rstrip()
        linedata = line.split("\t")
        xVals.append((float(linedata[0]),float(linedata[1])))
        yVals.append(float(linedata[2]))
    xMat = asmatrix(xVals)
    #plt.plot(xMat[:,1],yVals, 'rs')
    #plt.show()
    return xVals, yVals

def standRegres(xVal, yVal):
    xMat = mat(asarray(xVal))
    yMat = mat(asarray(yVal)).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) != 0.0:
        theta = xTx.I * (xMat.T * yMat)
        l = arange(0.,1.1,.1)
        #plt.plot(asarray(xMat[:,1]), asarray(yMat), 'gs', l, l*theta[1,0] + theta[0,0], 'r--')
        #plt.show()
        return theta
    else:
        print("Matrix is singular, trying gradient descent")
        return gradientDescent(xMat, yMat)

def polyRegres(xVal, yVal):
    xMat = asmatrix(xVal)
    yMat = asmatrix(yVal).T
    x2 = empty((len(xVal),3))
    for i in range(0, len(xVal)):
        x2[i,0] = xMat[i,0]
        x2[i,1] = xMat[i,1]
        x2[i,2] = xMat[i,1] ** 2
    xMat = asmatrix(x2)
    xTx = xMat.T * xMat
    if linalg.det(xTx) != 0.0:
        theta = xTx.I * (xMat.T * yMat)
        l = arange(0.,1.1,.1)
        #plt.plot(asarray(xMat[:,1]), asarray(yMat), 'gs', l, (l**2)*theta[2,0] + l*theta[1,0] + theta[0,0], 'r--')
        #plt.show()
        return theta
    else:
        print("Matrix is singular, can't find inverse")


def gradientDescent(xMat, yMat):
    alpha = .05
    n = yMat.size
    theta =asmatrix(zeros((xMat.shape[1],1)))#wrong size
    for i in range(0, 1000):
        hypothesis = xMat * theta
        loss = hypothesis - yMat
        squaredLoss = empty_like(loss)
        for el in range(0,loss.size):
            squaredLoss.flat[el] = loss.flat[el] ** 2
        cost = sum(squaredLoss) / 2
        gradient = (xMat.T * loss)
        theta = theta - alpha * gradient
    return theta

def fiveRegres(xVal, yVal):
    xMat = asmatrix(xVal)
    yMat = asmatrix(yVal).T
    x2 = empty((len(xVal),6))
    for i in range(0, len(xVal)):
        x2[i,0] = xMat[i,0]
        x2[i,1] = xMat[i,1]
        x2[i,2] = xMat[i,1] ** 2
        x2[i,3] = xMat[i,1] ** 3
        x2[i,4] = xMat[i,1] ** 4
        x2[i,5] = xMat[i,1] ** 5

    xMat = asmatrix(x2)
    xTx = xMat.T * xMat
    if linalg.det(xTx) != 0.0:
        theta = xTx.I * (xMat.T * yMat)
        l = arange(0.,1.1,.1)
        plt.plot(asarray(xMat[:,1]), asarray(yMat), 'gs', l, (l**5)*theta[5,0] + (l**4)*theta[4,0] + (l**3)*theta[3,0] + (l**2)*theta[2,0] + l*theta[1,0] + theta[0,0], 'r--')
        plt.show()
        return theta
    else:
        print("Matrix is singular, can't find inverse")

if __name__ == "__main__":
    main()