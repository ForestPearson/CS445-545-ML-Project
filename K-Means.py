from turtle import color, distance
from typing import Counter
import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

        self.kValues= self.kValues.reshape(self.kCount, 3, 1024)
        self.kValues=numpy.trunc(numpy.mean(self.kValues, axis=2))

   
        #Running the algorithm for the number of rounds, updating the K values to the mean of each cluster then cluster assignments
        #Keep track of past k values so we can plot the convergence and the final/best k values depending on the square error or however we judge it
       

        #Then plot.
        for i in range(self.rounds): 

           
            self.pastK.append(self.kValues) #Storing the initial k values
            self.clusters = self.clusterAssignments() # Assign initial clusters
            #  ~ Needs work 
            self.kValues = self.updateK()

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

            #converting self.clusters into array(length kcount), with each index containing all the data
            #points under a given cluster
            count=[]
            res=[]
            for i in range(self.kCount):
                temp=[]
                counter=0
                #for x in self.clusters:
                for x, v in enumerate(self.clusters):
                    if(v == i):
                        temp.append(self.testing_data[x])
                        counter=counter +1
                        
                #print(" at ", i , ":  ", temp)
                res.append(temp)
                count.append(counter)


            temp=self.clusters
            self.clusters=res
            self.plotter()
            self.clusters=temp
            
            #these scatter plot lines are just for testing until I can get my code working with the preexisting plotter function
            #AKA - can be deleted and replaced w plotter func. later
        return
    
    def clusterAssignments(self):#For the length of the data go through it, determin euclidean distance from each k value, and assign it to the closest k value
        result = [[] for i in range(self.kCount)] #List comprehension for dynamically assigning clusters
       
        print("kvalues")
        print(self.kValues.shape)

       # mean = numpy.mean(self.testing_data, axis = 0)
        #print(mean)
        #centroids = mean.reshape((1,3)) + (self.kValues.reshape(self.kCount,3))
        centroids=self.kValues
        print(centroids)
        centroids=centroids.astype(int)
        self.testing_data.astype(int)
 
        for i in self.testing_data:
            temp_distance=0
            distance=0
            clust = 0
            for x in range(len(centroids)):
   
                square = numpy.square(i - centroids[x])
                sum_square = numpy.sum(square)
                temp_distance = numpy.sqrt(sum_square)

                if(temp_distance < distance or distance == 0):
                    distance=temp_distance
                    clust= x

            result= numpy.append(result,clust)

            #result= numpy.append(result, numpy.argmin(numpy.sum((i.reshape((1,3))-centroids) ** 2, axis=0)))
       
        print("result")
        print(result)
        result.reshape(50000,1)

        self.kValues=centroids
        
        return result

        
    def updateK(self): #Moving the position of the k values to the centre of the mean of each cluster
        #result = [[] for i in range(self.kCount)]
        result=[]

        #for each cluster point, find the mean (center point) of that data - update list of clusters
        for i in range(self.kCount):
            count=0
            
            temp=[]
            for x in range (len(self.testing_data)):
                if(self.clusters[x] == i):
                    count=count+1                 
                    temp=numpy.append(temp,self.testing_data[x]) 
                    
            #if this cluster contains any nearby data points
            if(count):
                temp=temp.reshape(count, 3)
                mean = numpy.mean(temp, axis = 0)

            #otherwise were just going to re-use original cluser (fornow - need to fix clusterassignment function)
            else:
                mean=self.kValues[i]
 
            result=numpy.append(result, mean) 
                
        result=result.reshape(self.kCount,3)

        return result


    def plotter(self):
        #next 2 lines make it 3d - delete to make 2d again
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection ='3d')
        for i in range(self.kCount): #Plots a multi demensional graph of the clusters
            if(self.clusters[i]):

                temp = numpy.asarray(self.clusters[i])

                #to make 2d- delete next 2 lines + uncomment 2 below
                #ax.scatter(temp.T[0],temp.T[1], temp.T[2], cmap=plt.get_cmap(name=None, lut=None)) #Assigning colors to the clusters
                #ax.scatter(self.kValues.T[0],self.kValues.T[1], self.kValues.T[2], c='black')
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