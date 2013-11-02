#-*- coding: utf-8 -*-
from numpy import *
import operator

'''构造数据'''
def createDataSet():
    characters=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return characters,labels

'''从文件中读取数据，将文本记录转换为矩阵，提取其中特征和类标'''
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)        #得到文件行数
    returnMat=zeros((numberOfLines,3))     #创建以零填充的numberOfLines*3的NumPy矩阵
    classLabelVector=[]
    index=0
    for line in arrayOLines:              #解析文件数据到列表
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index, : ]=listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index+=1
    return returnMat,classLabelVector     #返回特征矩阵和类标集合

'''归一化数字特征值到0-1范围'''
'''输入为特征值矩阵'''
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals
    
def classify(sample,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]     #数据集行数即数据集记录数
    '''距离计算'''
    diffMat=tile(sample,(dataSetSize,1))-dataSet         #样本与原先所有样本的差值矩阵
    sqDiffMat=diffMat**2      #差值矩阵平方
    sqDistances=sqDiffMat.sum(axis=1)       #计算每一行上元素的和
    distances=sqDistances**0.5   #开方
    sortedDistIndicies=distances.argsort()      #按distances中元素进行升序排序后得到的对应下标的列表
    '''选择距离最小的k个点'''
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    '''从大到小排序'''
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

'''针对约会网站数据的测试代码'''
def datingClassTest():
    hoRatio=0.20          #测试样例数据比例
    datingDataMat,datingLabels=file2matrix('datingTestSet1.txt')
    normMat, ranges, minVals=autoNorm(datingDataMat)
    m =normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    k=4
    for i in range(numTestVecs):
        classifierResult=classify(normMat[i, : ],normMat[numTestVecs:m, : ], datingLabels[numTestVecs:m],k)
        print("The classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels [i] ) :
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def main():
    sample=[0,0]
    k=3
    group,labels=createDataSet()
    label=classify(sample,group,labels,k)
    print("Classified Label:"+label)

if __name__=='__main__':
    main()
    #datingClassTest()