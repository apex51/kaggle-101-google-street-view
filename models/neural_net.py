from PIL import Image
from numpy import *
from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne.updates import nesterov_momentum, sgd
from sklearn.metrics import accuracy_score

ORD_VALUES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

# Transform from string to index.  
def str_to_feature(st):
    return ORD_VALUES.index(ord(st))

# Transform from index to string
def feature_to_str(feature):
    return chr(ORD_VALUES[feature])

# Read images to ndarray under the given path.  
def read_image(path, imageIndex, imageSize):
    dataSet = []
    for index in imageIndex:
        fileName = '{}/{}.Bmp'.format(path, index)
        # Turn the 20*20 image into a (400,0) array.  
        imArr = list(Image.open(fileName, 'r').convert('L').getdata())
        dataSet.append(imArr)
    # Turn dataSet into a 0/1 matrix.  
    return (array(dataSet)/255).astype(float32)
    
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

# Pre-process feature to a (1, 62) vector.
def feature_to_vector(trainLabel, trainIndex):
    trainVec = zeros((len(trainIndex), len(ORD_VALUES)), dtype=float32)
    for (index, labelIndex) in enumerate(trainIndex):
        vecIndex = ORD_VALUES.index(ord(trainLabel[labelIndex]))
        trainVec[index, vecIndex] = 1.0
    return trainVec

# Post-process vector(1, 62) to feature, say a 0~61 num.
def vector_to_feature(predVec):
    return argmax(predVec, axis=1)




# The image size is 20*20 = 400.  
imageSize = 400

# Load data.  
trainLabel, trainIndex = read_csv('trainLabels.csv')
testLabel, testIndex = read_csv('testLabels.csv')
trainData = read_image('C:/Users/Victor/Desktop/M/train', trainIndex, imageSize)
testData = read_image('C:/Users/Victor/Desktop/M/test', testIndex, imageSize)
trainFeat = [str_to_feature(trainLabel[index]) for index in trainIndex]
trainVec = feature_to_vector(trainLabel, trainIndex)

print trainData.shape, trainVec.shape
print trainData.dtype, trainVec.dtype
print trainVec[1]

model = NeuralNet(layers=[('input', layers.InputLayer),
                          ('hidden', layers.DenseLayer),
                          ('output', layers.DenseLayer),],
    # layer parameters:
    input_shape=(None, 400),
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=62,  # 62 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=300,  # we want to train this many epochs
    verbose=1,)

model.fit(trainData, trainVec)
ytest_pred = model.predict(trainData)
featPred = vector_to_feature(ytest_pred)
print accuracy_score(featPred, trainFeat)
