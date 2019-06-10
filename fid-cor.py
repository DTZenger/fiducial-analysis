import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def get_means(x):
    return(x.mean())

class point:
    
    def __init__(self,data):
        self.data=data#nx2 array n is number of trials
        self.means=np.apply_along_axis(get_means,0,data)#1x2 array [x,y]
        print(self.means)

class Fids:
    
    def __init__(self,filename):
        self.filename=filename
        self.data=pd.read_csv(filename, sep=',',header=None)#read csv
        self.data=np.array(self.data)#Turn into numpy array
        self.data=np.delete(self.data,2,1)#delete z coordinate column - not needed and saves computation time

        
        self.points=[]#All points are stored in this array
        for i in range(4):
            x=self.data[i]

            for j in range(1,self.data.shape[0]//4):
                x=np.vstack((x,self.data[i+j*4]))
            self.points.append(point(x))
            x=np.array([])
            
        self.dist_data=np.zeros([4,4])

    #This function calculates the distance between two points
    def pointDist(self):
        for i in range(4):
            for j in range(i,4):
                self.dist_data[i][j]=self.dist_data[j][i]=np.linalg.norm(self.points[i].means-self.points[j].means)
    
    def angles(self):
        diff=self.points[0].means-self.points[3].means
        a1=np.arctan(diff[1]/diff[0])
        diff=self.points[2].means-self.points[1].means
        a2=np.arctan(diff[1]/diff[0])
        
        print("\n{} {}\n".format(a1,a2))
        return np.array([a1,a2])
        
    
    def cornerFidDist(self, cornerData):
        angle=np.mean(self.angles())
        rotMat=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        
        #v1=np.matmul(rotMat,self.points[0].means.T)
        #v2=np.matmul(rotMat,cornerData[0].T)
        #diff=abs(v2-v1)
        
        for i in range(4):
            v1=np.matmul(rotMat,self.points[i].means.T)
            v2=np.matmul(rotMat,cornerData[i].T)
            print(v1,v2)
            diff=abs(v2-v1)
            print(diff)

    def plotFid(self):
        plt.figure()
        """
        #Plot lines
        for i in range(4):
            for j in range(i,4):
                x1,x2=self.data_mean[i][0],self.data_mean[j][0]
                y1,y2=self.data_mean[i][1],self.data_mean[j][1]
                plt.plot([x1,x2],[y1,y2],'-k')
        """
        #plot points
        for i in range(4):
            plt.plot(self.points[i].means[0],self.points[i].means[1],'o')
    
        
        
        plt.show()
        
F=Fids('6-10-PatternMatchPosition-G.csv')
F.pointDist()
#F.angles()
print("\n")

data=pd.read_csv("6-10-Coords.csv", sep=',',header=None)#read csv
data=np.array(data)#Turn into numpy array
data=np.delete(data,2,1)#delete z coordinate column - not needed and saves computation time
print(data)
F.cornerFidDist(data)

#F.plotFid()