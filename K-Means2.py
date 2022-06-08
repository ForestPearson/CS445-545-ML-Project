from math import dist
from turtle import color
import numpy
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import cluster
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
    def __init__(self, k, rounds, data,label):
        self.testing_data = data/255
        self.testing_label = label
        self.kCount = k #Number of clusters
        self.rounds = rounds #Number of rounds to run the algorithm
        self.kValues = [] 
        self.clusters = [[] for i in range(self.kCount)] #Space for the clusters of k amount
        self.pastK = []
        self.sumOfSquares = []
        self.centroids=[]

    def run(self):
        
        self.data_list,self.features = self.testing_data.shape
		
        #pick random center
        random_center = numpy.random.choice(self.data_list,self.kCount,replace=False)	
        self.centroids = [self.testing_data[index]for index in random_center]

        for _ in range(self.rounds):
            self.clusters = self.get_clusters()
            #print(len(self.clusters[0]))
            #update new centroids
            old_centroids = self.centroids.copy()
            self.centroids = self.updated_centroids()

            self.calculate_accuracy()
            if(self.check_converged(old_centroids,self.centroids)):
                print("Converges")
                break
            #check if converged then stop loop
        
        return 0



            

				


    def get_clusters(self):
        clusters = [[]for _ in range(self.kCount)]

        for m,n in enumerate(self.testing_data):
            #between data and centroids
            dist  = [self.square(n,centroids) for centroids in self.centroids]
            index = numpy.argmin(dist)
            clusters[index].append(m)

        return clusters
            
    def square(self,m,n):#Squares by going through each k value and summing the square of the distance from each k value to the mean of the cluster, might want to do something else.
        return numpy.sqrt(numpy.sum((m-n)**2))

    def updated_centroids(self):
        new_centroids = numpy.zeros((self.kCount,self.features))

        for m, n in enumerate(self.clusters):
            mean = numpy.mean(self.testing_data[n],axis=0)
            new_centroids[m] = mean
		
        return new_centroids
		
    def check_converged(self,old_centroids,new_centroids):
        dist = [self.square(old_centroids[i],new_centroids[i]) for i in range(self.kCount)]
        if(sum(dist) == 0):
            return True
        else:
            return False

    
    def calculate_accuracy(self):
        account_label = [[0]* len(self.clusters) for i in range(len(self.clusters))]
        acc = [0]* len(self.clusters)
        for i in range(len(self.clusters)):
            for j in self.clusters[i]:
                account_label[i][self.testing_label[j]] += 1
        #calculate accuracy for each item
        for c in range(len(account_label)):
            max = numpy.max(account_label[c])
            total_data = numpy.sum(account_label[c])
            acc[c] = max/total_data
            

        #print(account_label)
        #print(acc)
        final_acc = numpy.sum(acc)/len(acc)
        print("**",final_acc)
        return 1
    
    def plotter(self):
        for i in range(self.kCount): #Plots a multi demensional graph of the clusters
            temp = numpy.asarray(self.clusters[i])
            plt.scatter(temp.T[0],temp.T[1], cmap=plt.get_cmap(name=None, lut=None)) #Assigning colors to the clusters
            plt.scatter(self.kValues.T[0],self.kValues.T[1], c='black')
        plt.show()
        plt.savefig('temp.png')
        return

    
    

if __name__ == '__main__':
    trainingData, trainingLabels = readTrainingData(FILEPATH)
    testingData, testingLabels = readTestingData(FILEPATH)
 
    kMeans = Kmeans(10,100,trainingData,trainingLabels)#Clusters, rounds, non-labeled data.
    kMeans.run()
    