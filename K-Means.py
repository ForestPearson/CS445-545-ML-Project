from turtle import color
import numpy
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.cluster import KMeans

'''GLOBALS'''
FILEPATH = "cifar-10-batches-py"

def unpickle(file):
    """Needed for loading the cifar data"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1') #may need to encode as 'latin1'
    return dict

def readTrainingData(pathToData):
    """Reads in the cifar data in batches, then returns images and labels as 2 arrays"""
    trainingImages = numpy.zeros([50000, 3072])
    trainingLabels = numpy.zeros([50000])

    count = 0
    imageQuantity = 10000 #per training batch
    for i in range(1, 6):
        batchPath = os.path.join(pathToData, "data_batch_{}".format(i))
        imageDict = unpickle(batchPath)
        trainingImages[count: count + imageQuantity, :] = imageDict["data"]
        trainingLabels[count: count + imageQuantity] = imageDict["labels"]
        count += imageQuantity
    return numpy.asarray(trainingImages, dtype = numpy.int), numpy.asarray(trainingLabels, dtype = numpy.int)

def readTestingData(pathToData):
    """Reads in the cifar test batch"""
    batchPath = os.path.join(pathToData, "test_batch")
    imageDict = unpickle(batchPath)
    testingImages = imageDict["data"]
    testingLabels = imageDict["labels"]

    #dataset = numpy.load('cifar-10-batches-py/test_batch', allow_pickle = True, encoding = 'bytes')

    return numpy.asarray(testingImages, dtype = numpy.int), numpy.asarray(testingLabels, dtype = numpy.int)


class Kmeans():
    def __init__(self, k, rounds, data):
        self.testing_data = data
        self.kCount = k #Number of clusters
        self.rounds = rounds #Number of rounds to run the algorithm
        self.kValues = [] 
        self.clusters = [[] for i in range(self.kCount)] #Space for the clusters of k amount
        self.pastK = []
        self.sumOfSquares = []
        self.centroids=[]

    def run(self):
        print(self.testing_data.shape)
        kStarter = [] #Space for initial set of k values
        index = numpy.random.choice(len(self.testing_data), self.kCount, replace = False) #Choosing k random positions in the input space, 9999 needs to be changed to the length of the data.
        for i in index:
            kStarter.append(self.testing_data[i])


        self.kValues = numpy.asarray(kStarter) #You can convert to a numpy array after appending, since appending is a pain in numpy, if you figure out it I'll be amazed.
    
        #reshaping each image to show the mean for each RBG color
        color_divided= self.testing_data.reshape(50000, 3, 1024)
        color_divided_temp = numpy.trunc(numpy.mean(color_divided, axis=2))
        self.testing_data =color_divided_temp

        self.kValues= self.kValues.reshape(5, 3, 1024)
        self.kValues=numpy.trunc(numpy.mean(self.kValues, axis=2))
   
        #Running the algorithm for the number of rounds, updating the K values to the mean of each cluster then cluster assignments
        #Keep track of past k values so we can plot the convergence and the final/best k values depending on the square error or however we judge it
       
        self.rounds = 1

        #Then plot.
        for i in range(self.rounds): 

            self.clusters = self.clusterAssignments() # Assign initial clusters
            self.pastK.append(self.kValues) #Storing the initial k values

            #  ~ Needs work 
            #self.kValues = self.updateK()

            #these scatter plot lines are just for testing until I can get my code working with the preexisting plotter function
            #AKA - can be deleted and replaced w plotter func. later
            plt.scatter(self.testing_data.T[0],self.testing_data.T[1], cmap=plt.get_cmap(name=None, lut=None)) #Assigning colors to the clusters
            plt.scatter(self.kValues.T[0],self.kValues.T[1], c='black')
            plt.show()
      
          

        #Judge the quality of the final k values



        #Print data
        print("each image's mean color values : ")
        print(self.testing_data)
        print(self.testing_data.shape)

        print("kvalues : ")
        print(self.kValues)
        print(self.kValues.shape)

        #output shows a lot of/ all items in the same cluster- maybe once update k will help adjust?
        print ("clusters from clusterAssignments  :")
        print(self.clusters) 
        print(len(self.clusters))

        return
    
    def clusterAssignments(self):#For the length of the data go through it, determin euclidean distance from each k value, and assign it to the closest k value
        result = [[] for i in range(self.kCount)] #List comprehension for dynamically assigning clusters
       
        mean = numpy.mean(self.testing_data, axis = 0)
        centroids = mean.reshape((1,3)) + self.kValues

        #print statments can be deleted - just for testing
        print("centroids: ")
        print(centroids)
 
        for i in self.testing_data:
            result= numpy.append(result, numpy.argmin(numpy.sum((i.reshape((1,3))-centroids) ** 3, axis=1)))
       
        result.reshape(50000, 1)
        return result

        
    def updateK(self): #Moving the position of the k values to the centre of the mean of each cluster
        result = [[] for i in range(self.kCount)]

        #what I have so far but currently not working - still trying but if anyone has suggestions
        for i in range(len(self.clusters)):
            result= numpy.append(result, numpy.mean([self.testing_data[x] for x in range(len(self.testing_data)) if self.clusters[x] == i], axis=0))

        return self.kValues
    
    def plotter(self):
        for i in range(self.kCount): #Plots a multi demensional graph of the clusters
            temp = numpy.asarray(self.clusters[i])
            plt.scatter(temp.T[0],temp.T[1], cmap=plt.get_cmap(name=None, lut=None)) #Assigning colors to the clusters
            plt.scatter(self.kValues.T[0],self.kValues.T[1], c='black')
        plt.show()
        plt.savefig('temp.png')
        return

    
    def square(self):#Squares by going through each k value and summing the square of the distance from each k value to the mean of the cluster, might want to do something else.
        return 0

if __name__ == '__main__':
    trainingData, trainingLabels = readTrainingData(FILEPATH)
    testingData, testingLabels = readTestingData(FILEPATH)
 
    kMeans = Kmeans(5,5,trainingData)#Clusters, rounds, non-labeled data.
    kMeans.run()