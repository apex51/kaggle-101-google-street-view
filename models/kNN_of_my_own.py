from PIL import Image
from numpy import *
from sklearn.cross_validation import train_test_split
from numpy import linalg as la
from skimage.io import imread
from skimage.feature import hog

# # Read images to mat under the given path.  
# def read_image(path, imageIndex, imageSize):
#     dataSet = []
#     for index in imageIndex:
#         fileName = '{}/{}.Bmp'.format(path, index)
#         # Turn the 20*20 image into a (400,0) array.  
#         imArr = list(Image.open(fileName, 'r').convert('L').getdata())
#         dataSet.append(imArr)
#     # Turn dataSet into a 0/1 matrix.  
#     return mat(dataSet)
    

# # Read images to mat under the given path.  
# def read_image(path, imageIndex, imageSize):
#     dataSet = []
#     for index in imageIndex:
#         fileName = '{}/{}.Bmp'.format(path, index)
#         # Turn the 20*20 image into a (400,0) array.  
#         imArr = list(Image.open(fileName, 'r').convert('L').getdata())
#         imArr = (imArr - mean(imArr))/std(imArr)
#         dataSet.append(imArr)
#     # Turn dataSet into a 0/1 matrix.  
#     return mat(dataSet)

# Read hog of image to mat under the given path.  
def read_image(path, imageIndex, imageSize):
    dataSet = []
    for index in imageIndex:
        fileName = '{}/{}.Bmp'.format(path, index)
        image = imread(fileName, as_grey=True)
        fd = hog(image, orientations=8, pixels_per_cell=(7, 7), cells_per_block=(1, 1))
        dataSet.append(fd)
    # Turn dataSet into a 0/1 matrix.  
    return mat(dataSet)


# Read csv to dict(image index:label) and list(image index) given the file dir.  
def read_csv(fileName):
    retDict = {}
    retArr = []
    fr = open(fileName)
    for line in fr.readlines()[1:]:
        curLine = line.strip().split(',')
        retDict[int(curLine[0])] = curLine[1]
        retArr.append(int(curLine[0]))
    return retDict, retArr

# # k-NN kernel using Eucledian Dist
# def classify(imageVec, dataMat, imageLabel, imageIndex, k):
#     dataSize = shape(dataMat)[0]
#     # print 'the dataMat shape is {}'.format(shape(dataMat))
#     diffMat = mat(tile(imageVec, (dataSize,1))) - dataMat
#     # print 'the diffMat shape is {}'.format(shape(diffMat))
#     distMat = sum(power(diffMat, 2), axis=1)
#     # print 'the distMat shape is {}'.format(shape(distMat))
#     indexArr = argsort(distMat.A.T[0])
#     labelCount = {}
#     for i in range(0,k):
#         iLabel = imageLabel[imageIndex[indexArr[i]]]
#         labelCount[iLabel] = labelCount.get(iLabel, 0) + 1
#     bestLabel = sorted(labelCount.items(), key=lambda p:p[1], reverse=True)[0][0]
#     return bestLabel


# k-NN kernel using cosine distance
def classify(imageVec, dataMat, imageLabel, imageIndex, k):
    distArr = []
    for dataVec in dataMat:
        num = float(dot(imageVec,dataVec))
        denorm = la.norm(imageVec) * la.norm(dataVec)
        distArr.append(num/denorm)
    indexArr = argsort(distArr)[::-1]
    labelCount = {}
    for i in range(0,k):
        iLabel = imageLabel[imageIndex[indexArr[i]]]
        labelCount[iLabel] = labelCount.get(iLabel, 0) + 1
    bestLabel = sorted(labelCount.items(), key=lambda p:p[1], reverse=True)[0][0]
    return bestLabel

# # k-NN kernel using corrcoef
# def classify(imageVec, dataMat, imageLabel, imageIndex, k):
#     distArr = []
#     for dataVec in dataMat:
#         distArr.append(corrcoef(imageVec, dataVec)[0,1])
#     # print distArr
#     indexArr = argsort(distArr)[::-1]
#     labelCount = {}
#     for i in range(0,k):
#         iLabel = imageLabel[imageIndex[indexArr[i]]]
#         labelCount[iLabel] = labelCount.get(iLabel, 0) + 1
#     bestLabel = sorted(labelCount.items(), key=lambda p:p[1], reverse=True)[0][0]
#     return bestLabel

# The image size is 20*20 = 400.  
imageSize = 100

# Load data.  
trainLabel, trainIndex = read_csv('trainLabels.csv')
testLabel, testIndex = read_csv('testLabels.csv')
trainData = read_image('C:/Users/Victor/Desktop/M/train50', trainIndex, imageSize)
# testData = read_image('C:/Users/Victor/Desktop/M/test10', testIndex, imageSize)


# trainDataA, trainDataB, trainIndexA, trainIndexB = train_test_split(trainData, trainIndex, test_size = 0.1)
# # The training process.  
# dataSizeB = shape(trainDataB)[0]
# numRight = 0.0
# for i in range(dataSizeB):
#     result = classify(trainDataB[i], trainDataA, trainLabel, trainIndexA, 1)
#     if result == trainLabel[trainIndexB[i]]:
#         numRight += 1
# accuracyRate = numRight/dataSizeB
# print accuracyRate

# 5-fold CV
accuracyRate = []
for i in range(5):
    trainDataA, trainDataB, trainIndexA, trainIndexB = train_test_split(trainData, trainIndex, test_size = 0.1, random_state=1)
    dataSizeB = shape(trainDataB)[0]
    numRight = 0.0
    for i in range(dataSizeB):
        result = classify(trainDataB[i], trainDataA, trainLabel, trainIndexA, 1)
        if result == trainLabel[trainIndexB[i]]:
            numRight += 1
    accuracyRate.append(numRight/dataSizeB)
print mean(accuracyRate)

# # Let's try on the test data.  
# fw = open('output-corrcoef.csv', 'w')
# dataSize = shape(testData)[0]
# csvStr = 'ID,Class\n'
# for i in range(dataSize):
#     result = classify(testData[i], trainData, trainLabel, trainIndex, 1)
#     csvStr += '{},{}\n'.format(testIndex[i],result)
# fw.write(csvStr)
# fw.close()
