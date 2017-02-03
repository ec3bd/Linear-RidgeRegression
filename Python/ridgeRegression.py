#Eamon Collins
#ec3bd
#Tested using python 3.4.

from numpy import *
import matplotlib.pyplot as plt
import sys

def main():
    xVals, yVals = loadDataset("RRdata.txt")
    #print(ridgeRegress(xVals, yVals))
    ridge = cv(xVals, yVals)
    print(ridgeRegress(xVals, yVals, ridge))
    betaRR = ridgeRegress(xVals, yVals, ridge)
    scatterPlotPlane(xVals, yVals, betaRR)  #can't get to work
    print(ridgeRegress(xVals, yVals, 0)) #for 1.5
    print(ridgeRegress(xVals[:,1], xVals[:,2],0))

def ridgeRegress(xVal, yVal, ridge = 0):
    #used ridge instead of lambda as lambda is a keyword
    xTx = xVal.T * xVal
    if linalg.det(xTx) != 0.0:
        ridgedI = ridge*eye(xTx.shape[1])
        ridgedI[0,0] = 0 #so as to not regularize B0
        plusridge = asmatrix(xTx + ridgedI)
        return plusridge.I * (xVal.T * yVal)

def cv(xVal, yVal):
    n = xVal.shape[0]
    p = xVal.shape[1]
    augmented = concatenate((xVal,yVal),1)
    random.seed(37)
    random.shuffle(augmented)
    augmented = asmatrix(augmented)
    xVal = augmented[:, 0:p]
    yVal = augmented[:, p]
    bestError = sys.maxsize
    bestridge = 0
    for j in range(0,50):
        ridge = (j+1) * .02
        foldMSE = []
        for i in range(0,10):
            beg = int(i*n/10)
            end = int((i+1)*n/10)
            xTest = xVal[beg:end,:]
            xTrain = xVal[0:beg:1,:]
            xTrain2 = xVal[end:,:]
            xTrain = concatenate((xTrain, xTrain2))
            yTest = yVal[beg:end,:]
            yTrain = yVal[0:beg:1,:]
            yTrain2 = yVal[end:,:]
            yTrain = concatenate((yTrain, yTrain2))

            Bi = ridgeRegress(xTrain, yTrain, ridge)
            hypothesis = xTest * Bi
            error = yTest - hypothesis
            squaredError = empty_like(error)
            for el in range(0,error.size):
                squaredError.flat[el] = error.flat[el] ** 2
            foldMSE.append(sum(squaredError) / yTest.shape[0]) #don't know whether to do MSE or just mean and where
        meanError = sum(foldMSE) / len(foldMSE)
        if meanError < bestError:
            bestError = meanError
            bestridge = ridge
    print(bestError)
    print(bestridge)

    return bestridge

def scatterPlotPlane(xVal, yVal, beta):
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # dont know why this doesn't work
    plt.hold(True)

    # x_surf = np.sort(xVal[:,1])  # generate a mesh
    # y_surf = np.sort(xVal[:,2])
    x_surf, y_surf = meshgrid(xVal[:,1], xVal[:,2])
    z_surf = (dot(xVal, beta))  # ex. function, which depends on x and y
    ax.plot_surface(x_surf, y_surf, z_surf)  # plot a 3d surface plot




    ax.scatter(xVal[:,1], xVal[:,2], yVal, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    return

def loadDataset(filename):
    file = open(filename, 'r')
    xVals = []
    yVals = []
    for line in file:
        line = line.rstrip()
        linedata = line.split(" ")
        xVals.append((float(linedata[0]),float(linedata[1]), float(linedata[2])))
        yVals.append(float(linedata[3]))
    xVals = asmatrix(xVals)
    yVals = asmatrix(yVals).T
    return xVals, yVals

if __name__ == "__main__":
    main()