from numpy import *
import matplotlib.pylab as plt



def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(2):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#需要求解析解 参数=（（X的转置 * X）的逆）* X的转置 * 因变量向量
def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0: #矩阵存在逆的条件是行列式不为0
        return
    
    ws = xTx.I * (xMat.T * yMat)
    return ws

def regression1():
    xArr, yArr = loadDataSet("GANtest/lesson 1/data.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)               #add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax.scatter(xMat[:, 1].flatten().tolist(), yMat.T[:, 0].flatten().A[0].tolist()) #scatter 的x是xMat中的第二列，y是yMat的第一列
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

regression1()