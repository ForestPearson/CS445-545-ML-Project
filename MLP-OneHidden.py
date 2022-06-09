import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
np.set_printoptions(threshold=sys.maxsize)

HIDDENUNITS = 100
LEARNINGRATE = 0.01
EPOCHS = 100
MOMENTUM = 0.9
BIAS = 1
cifarClasses = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
dataset1 = np.load('cifar-10-batches-py/data_batch_1', allow_pickle=True, encoding='bytes')
dataset2 = np.load('cifar-10-batches-py/data_batch_2', allow_pickle=True, encoding='bytes')
dataset3 = np.load('cifar-10-batches-py/data_batch_3', allow_pickle=True, encoding='bytes')
dataset4 = np.load('cifar-10-batches-py/data_batch_4', allow_pickle=True, encoding='bytes')
dataset5 = np.load('cifar-10-batches-py/data_batch_5', allow_pickle=True, encoding='bytes')
testset = np.load('cifar-10-batches-py/test_batch', allow_pickle=True, encoding='bytes')

trainData = np.concatenate((dataset1[b'data'], dataset2[b'data'], dataset3[b'data'], dataset4[b'data'], dataset5[b'data']), axis=0)
trainLabels = np.concatenate((dataset1[b'labels'], dataset2[b'labels'], dataset3[b'labels'], dataset4[b'labels'], dataset5[b'labels']), axis=0)
trainData = trainData / 255

testData = testset[b'data']
testData = testData / 255
testLabels = testset[b'labels']

testingConfusionLabels = []
testingConfusionResults = []

def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

def randWeights(num):
    weight = []
    weight = np.random.uniform(-0.05, 0.05, num)
    return weight

def randWeightMatrix(n,m):
    weightMatrix = []
    for i in range(0,n):
        weightMatrix.append(randWeights(m))
    return weightMatrix

def runEpoch(inToHidden, hiddenToOut, prevIn, prevOut, learningRate, momentum, training):
    totalCorrect = 0

    for i in range(0, len(trainData)):
        inputs = trainData[i] 
        inputs = np.insert(inputs, [0], [BIAS], axis=0)
        inputs = inputs.reshape(1, len(inputs)) 
        label = trainLabels[i]

        # Forward Propagation
        hiddenInputs = np.dot(inputs, inToHidden)
        hiddenOutputs = sigmoid(hiddenInputs)
        #hiddenOutputs = np.insert(hiddenOutputs, [0], [BIAS], axis=1)

        #hiddenOutputs = hiddenOutputs.reshape(1, len(hiddenOutputs[0]))

        outInputs = np.dot(hiddenOutputs, hiddenToOut)
        outputs = sigmoid(outInputs)
        #outputs = outputs.reshape(1, len(outputs[0]))

        # Backward Propagation
        targetOutputs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        targetOutputs[label] = 0.9

        prediction = np.argmax(outputs) 
        if prediction == label:
            totalCorrect += 1
        if training:
            outputError = outputs * (1 - outputs) * (targetOutputs - outputs)
            hiddenError = hiddenOutputs * (1 - hiddenOutputs) * np.dot(outputError, np.transpose(hiddenToOut))

            deltaInput = (learningRate * hiddenError * (inputs.T)) + (momentum * prevIn)
            prevIn = deltaInput
            inToHidden += deltaInput

            deltaHidden = (learningRate * (outputError * np.transpose(hiddenOutputs))) + (momentum * prevOut)
            prevOut = deltaHidden
            hiddenToOut += deltaHidden

        else:
            testingConfusionLabels.append(label)
            testingConfusionResults.append(prediction)

    return (totalCorrect / len(trainData)) * 100, inToHidden, hiddenToOut, prevIn, prevOut
def printOne(Chosen, label, directory):
    img = trainData
    dataLabel = trainLabels[Chosen]
    img = img.reshape(50000,3,32,32).transpose(0,2,3,1)
    #plt.imshow(img[Chosen:Chosen+1][0])
    #plt.savefig("Results/"+str(directory)+"/"+str(label) + '.png')
    return img[Chosen:Chosen+1][0]
    #print(trainData[0])
    #print(cifarClasses[dataLabel])

def printClass(selected,sample):
    c = 0;
    acc = 0;
    plt.figure(figsize=(5,5))
    while(c < sample):
        plt.subplot(5,5,c+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        x = np.random.randint(len(testingConfusionResults))
        if(testingConfusionResults[x] == selected and c < 25):
            
            label = "Predicted"+str(selected)+"I"+str(c)+"_Class"+str(testingConfusionLabels[x])
            plt.imshow(printOne(x,label,selected))
            plt.xlabel(cifarClasses[testingConfusionLabels[x]])
            #printOne(x,label, selected)
            c += 1
            if(testingConfusionLabels[x] == testingConfusionResults[x]):
                acc += 1
    #for i in range(0, len(testingConfusionLabels)):
    #   if(testingConfusionResults[i] == selected and c < 10):
    #        label = "Predicted"+str(selected)+"I"+str(c)+"_Class"+str(testingConfusionLabels[i])
    #        printOne(i,label, selected)
    #        c += 1
    #        if(testingConfusionLabels[i] == testingConfusionResults[i]):
    #            acc += 1
    plt.savefig("Results/Class"+str(selected)+".png")
    print("Accuracy for class "+str(selected)+", "+str(cifarClasses[selected])+": "+str(acc/c))
            
def experiment1():
    printOne(1, "test", 42)
    inToHidden = randWeightMatrix(3073, HIDDENUNITS + 1)
    hiddenToOut = randWeightMatrix(HIDDENUNITS + 1, 10)
    prevIn = np.zeros((3073, HIDDENUNITS + 1))
    prevOut = np.zeros((HIDDENUNITS + 1, 10))

    for i in range(0, 1):
        testingConfusionLabels.clear()
        testingConfusionResults.clear()
        accuracy, inToHidden, hiddenToOut, prevIn, prevOut = runEpoch(inToHidden, hiddenToOut, prevIn, prevOut, LEARNINGRATE, MOMENTUM, True)
        print("Epoch: " + str(i) + " Accuracy: " + str(accuracy))
        accuracy, inToHidden, hiddenToOut, prevIn, prevOut = runEpoch(inToHidden, hiddenToOut, prevIn, prevOut, LEARNINGRATE, MOMENTUM, False)
        print("Epoch: " + str(i) + " Accuracy: " + str(accuracy))
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    printClass(0,25)
    printClass(1,25)
    printClass(2,25)
    printClass(3,25)
    printClass(4,25)
    printClass(5,25)
    printClass(6,25)
    printClass(7,25)
    printClass(8,25)
    printClass(9,25)
        
experiment1() 