import numpy
import pandas
from sklearn.metrics import confusion_matrix


class Perceptron():
    def __init__(self, epochs, momentum, partition, hiddenUnits):#Load and initialize data with the given parameters
        self.trainingFile = "???"
        self.testFile = "???"
        self.hiddenUnits = hiddenUnits
        self.learning = 0.1
        self.bias = 1
        self.epochs = epochs
        self.momentum = momentum
        self.confusionLabels = [] #Used to create the confusion matrix when testing
        self.confusionMatrix = []
        self.partition = partition
        self.inputWeight = "???" #Initialize all  the weights with random/zero values
        self.outputWeight = "???"
        self.pastInput = "???"
        self.pastHidden = "???"
    # Generates random weights uniformly distributed between -0.05 and 0.05

    def weights(self, n, m):
        return
    # This is the activation function for each neuron fired in the network.

    def sigmoid(self, n):
        return 1 / (1 + numpy.exp(-n))
    # Loading image data from the dataset given

    def loadImage(self, row, train):
        inputs = []

        return inputs
    # Loading label data from the dataset given to get the expected prediction

    def loadLabels(self, row, train):
        label = 0

        return label

    def epoch(self, train):
        #global inputWeight, outputWeight, pastInput, pastHidden
        totalRows = 0
        totalCorrect = 0
        if train:
            totalRows = "???"
        else:
            totalRows = "???"
        # Total rows, one qauter or one half of the dataset
        for row in range(int(totalRows/self.partition)):
            label = "???"
            # Gather pixels and bias then reshape to avoid mismatches
            "???"
            # perform dot product on the inputs and the input weights then the same for the hidden variables
            "???"
            # Find and check the latest prediction by looking at the largest output. Then create the adjusted label array to work with sigmoid.
            "???"
            adjustedLabel = numpy.zeros(10)
            adjustedLabel.fill(0.1)
            prediction = "???"#argmax of outputsigmoid
            if prediction == label:
                totalCorrect += 1
                adjustedLabel[label] = 0.9
            if train:
                # calculate the error for the output layer and then update the weights via back propagation
                "???"

            else:
                #Adding data for the confusion matrix
                self.confusionMatrix.append(prediction)
                self.confusionLabels.append(label)
        acc = totalCorrect / (totalRows/self.partition)
        return acc

    def training(self):
        for i in range(0, self.epochs):
            print("Epoch: " + str(i))
            trainingAccuracy = self.epoch(True)
            print("Training Accuracy: " + str(trainingAccuracy * 100))
            testingAccuracy = self.epoch(False)
            print("Testing Accuracy: " + str(testingAccuracy * 100))
            print("")
        matrix = confusion_matrix(self.confusionLabels, self.confusionMatrix)
        print(matrix)
