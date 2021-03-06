from PIL import Image
from numpy import *
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from skimage.io import imread
from skimage.feature import hog

ORD_VALUES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

# Transform from string to index.  
def str_to_feature(st):
    return ORD_VALUES.index(ord(st))

# Transform from index to string
def feature_to_str(feature):
    return chr(ORD_VALUES[feature])

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
    
# # Read corrcoef images to mat under the given path.  
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
        fd = hog(image, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1))
        fd = (fd - mean(fd))/std(fd)
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

# The image size is 20*20 = 400.  
imageSize = 400

# Load data.  
trainLabel, trainIndex = read_csv('trainLabels.csv')
# labelCount = {}
# for index in trainIndex:
#     labelCount[trainLabel[index]] = labelCount.get(trainLabel[index], 0) + 1
# countList = sorted(labelCount.iteritems(),key=lambda p:p[0])
# print countList

testLabel, testIndex = read_csv('testLabels.csv')
trainData = read_image('C:/Users/Victor/Desktop/M/train40', trainIndex, imageSize)
# testData = read_image('C:/Users/Victor/Desktop/M/test50', testIndex, imageSize)
trainFeat = [str_to_feature(trainLabel[index]) for index in trainIndex]

trainX, testX, trainY, testY = cross_validation.train_test_split(trainData, trainFeat, test_size=0.1)


model = ExtraTreesClassifier(n_estimators=1000,n_jobs=-1,random_state=1)

model.fit(trainX, trainY)
print model.score(testX, testY)

# # Let's try on the test data.
# model = ExtraTreesClassifier(n_estimators=1000,n_jobs=-1,random_state=1)
# model.fit(trainData, trainFeat)
# fw = open('output-ET-Corrcoef.csv', 'w')
# fw.write('ID,Class\n')
# dataSize = shape(testData)[0]
# predFeat = model.predict(testData)
# for i in range(dataSize):
#     fw.write('{},{}\n'.format(testIndex[i],feature_to_str(predFeat[i])))
# fw.close()