import numpy
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self, k, rounds, data):
        self.testing_data = data
        self.kCount = k #Number of clusters
        self.rounds = rounds #Number of rounds to run the algorithm
        self.kValues = [] 
        self.clusters = [[] for i in range(self.kCount)] #Space for the clusters of k amount
        self.pastK = []
        self.sumOfSquares = []
    def run(self):
        kStarter = [] #Space for initial set of k values
        index = numpy.random.choice(9999, self.kCount, replace = False) #Choosing k random positions in the input space, 9999 needs to be changed to the length of the data.
        for i in index:
            kStarter.append(self.testing_data[i])
        self.kValues = numpy.asarray(kStarter) #You can convert to a numpy array after appending, since appending is a pain in numpy, if you figure out it I'll be amazed.
        self.clusters = self.clusterAssignments() # Assign initial clusters
        self.pastK.append(self.kValues) #Storing the initial k values
        #Running the algorithm for the number of rounds, updating the K values to the mean of each cluster then cluster assignments
        #Keep track of past k values so we can plot the convergence and the final/best k values depending on the square error or however we judge it
        #Then plot.
        for i in range(self.rounds): 
            print("Todo")
        #Judge the quality of the final k values
        #Print data
        return
    
    def clusterAssignments(self):#For the length of the data go through it, determin euclidean distance from each k value, and assign it to the closest k value
        result = [[] for i in range(self.kCount)] #List comprehension for dynamically assigning clusters
        return result
    
    def plotter(self):
        for i in range(self.kCount): #Plots a multi demensional graph of the clusters
            temp = numpy.asarray(self.clusters[i])
            plt.scatter(temp.T[0],temp.T[1], cmap=plt.get_cmap(name=None, lut=None)) #Assigning colors to the clusters
            plt.scatter(self.kValues.T[0],self.kValues.T[1], c='black')
        plt.savefig('temp.png')
        return
    def updateK(self): #Moving the position of the k values to the centre of the mean of each cluster
        for i in range(self.kCount): #Calculating the mean of each cluster and set each K
            print("Todo")
        return self.kValues
    
    def square(self):#Squares by going through each k value and summing the square of the distance from each k value to the mean of the cluster, might want to do something else.
        return 0
data = ""
kMeans = Kmeans(5,5,data)#Clusters, rounds, non-labeled data.
kMeans.run()