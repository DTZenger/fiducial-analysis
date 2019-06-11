import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from uncertainties import unumpy as unp
from uncertainties import ufloat

def get_means(x):
    return(ufloat(x.mean(),0))
    #return(ufloat(x.mean(),x.std()/np.sqrt(5)))
    
#def get_ster(x):
    #return(x.std()/np.sqrt(5))



class point:
    
    def __init__(self,data):
        self.data=data#nx2 array n is number of trials
        self.means=np.apply_along_axis(get_means,0,data)#1x2 array [x,y]
        #self.ster=np.apply_along_axis(get_ster,0,data)

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
            
        self.dist_data=np.array([[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)],[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)],[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)],[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)]])
        self.dist_data_error=np.array([[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)],[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)],[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)],[ufloat(0,0),ufloat(0,0),ufloat(0,0),ufloat(0,0)]])

    #This function calculates the distance between two points
    def pointDist(self):
        for i in range(4):
            for j in range(i+1,4):
                #self.dist_data[i][j]=self.dist_data[j][i]=unp.linalg.norm(self.points[i].means-self.points[j].means)
                x1=self.points[i].means[0]
                #x1_err=self.points[i].ster[0]
                x2=self.points[j].means[0]
                #x2_err=self.points[j].ster[0]
                y1=self.points[i].means[1]
                #y1_err=self.points[i].ster[1]
                y2=self.points[j].means[1]
                #print(x1,x2,y1,y2)
                self.dist_data[i][j]=unp.sqrt((x1-x2)**2+(y1-y2)**2)
                #y2_err=self.points[j].ster[1]
                #num=((x1-x2)**2)*(x1_err**2+x2_err**2)+((y1-y2)**2)*(y1_err**2+y2_err**2)
                #den=(x1-x2)**2+(y1-y2)**2
                #self.dist_data_error[i][j]=np.sqrt(num/den)
    
    def printPointDist(self):
        print("Distances Between:\n")
        print("0 & 3: " + str(self.dist_data[0][3]))# + " +/- " + str(self.dist_data_error[0][3]))
        print("1 & 2: " + str(self.dist_data[1][2]))# + " +/- " + str(self.dist_data_error[1][2]))
        print()
        print("0 & 1: " + str(self.dist_data[0][1]))# + " +/- " + str(self.dist_data_error[0][1]))
        print("2 & 3: " + str(self.dist_data[2][3]))# + " +/- " + str(self.dist_data_error[2][3]))
        print()
        print("0 & 2: " + str(self.dist_data[0][2]))# + " +/- " + str(self.dist_data_error[0][2]))
        print("1 & 3: " + str(self.dist_data[1][3]))# + " +/- " + str(self.dist_data_error[1][3]))
    
    def angles(self):
        diff=self.points[0].means-self.points[3].means
        a1=unp.arctan(diff[1]/diff[0])
        diff=self.points[2].means-self.points[1].means
        a2=unp.arctan(diff[1]/diff[0])
        
        print("\n{} {}\n".format(a1,a2))
        return np.array([a1,a2])
        
    
    def cornerFidDist(self, cornerData):
        angle=np.mean(self.angles())
       # rotMat=np.array([[unp.cos(angle),unp.sin(angle)],[-unp.sin(angle),unp.cos(angle)]])
        
        #v1=np.matmul(rotMat,self.points[0].means.T)
        #v2=np.matmul(rotMat,cornerData[0].T)
        #diff=abs(v2-v1)
        
        for i in range(4):
            #v1=unp.matmul(rotMat,self.points[i].means.T)
            #v2=unp.matmul(rotMat,cornerData[i].T)
            v1=np.array([self.points[i].means[0]*unp.cos(angle)+self.points[i].means[1]*unp.sin(angle),
                          -self.points[i].means[0]*unp.sin(angle)+self.points[i].means[1]*unp.cos(angle)])
            v2=np.array([cornerData[i][0]*unp.cos(angle)+cornerData[i][1]*unp.sin(angle),
                -cornerData[i][0]*unp.sin(angle)+cornerData[i][1]*unp.cos(angle)])
            diff=abs(v2-v1)
            print(diff)
            #print(abs(diff-np.array([.1275,.1535])))

    def plotFid(self,point_data):
        plt.figure()

        #plot fiducial points
        x=np.array([])
        y=np.array([])
        for i in self.points:
            x=np.append(x,i.means[0].nominal_value)
            y=np.append(y,i.means[1].nominal_value)
            
        plt.plot(x,y,'o',label='Fiducial Points')
        
        #now plot corner points
        xc,yc=point_data.T
        plt.plot(xc,yc,'r^',label='Corner')
        plt.legend()
        
        plt.show()
        
F=Fids('6-11-PatternMatchPosition-F.csv')
F.pointDist()
F.printPointDist()
#F.angles()
print("\n")

data=pd.read_csv("6-11-Coords.csv", sep=',',header=None)#read csv
data=np.array(data)#Turn into numpy array
data=np.delete(data,2,1)#delete z coordinate column - not needed and saves computation time
print(data)
F.cornerFidDist(data)

F.plotFid(data)