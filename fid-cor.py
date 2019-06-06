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

F=Fids('6-5-PatternMatchPosition-F.csv')
F.pointDist()