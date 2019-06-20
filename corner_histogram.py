import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import stats

def get_val_diff_avg(x):
    vals=np.array([])
    x.flatten()
    for i in range(x.shape[0]//4):
        vals=np.append(vals,x[4*i]-x[4*i+3])
        vals=np.append(vals,x[4*i+1]-x[4*i+2])
    return np.mean(vals)

def rotate_values(x,angle):
    x=x.T
    
    angle=-angle
    
    rotated_values=np.matmul(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]),x)

    return (rotated_values.T)


class point_histogram:
    
    
    def __init__(self,filename):
        self.filename=filename
        self.data=pd.read_csv(filename, sep=',',header=None)#read csv
        self.data=np.array(self.data)#Turn into numpy array
        self.data=np.delete(self.data,2,1)#delete z coordinate column - not needed and saves computation time

    #This way approaches by averaging all the distances and then finds the angle
    def find_angle(self,data):
        values=np.apply_along_axis(get_val_diff_avg,0,data)
        return np.arctan(values[1]/values[0])
    #This way finds the angle by averaging all the angles at the end
    def find_angle2(self):
        angles=np.array([])
        for i in range(self.data.shape[0]//4):
            x=self.data[4*i]-self.data[4*i+3]
            angles=np.append(angles,np.arctan(x[1]/x[0]))
            y=self.data[4*i+1]-self.data[4*i+2]
            angles=np.append(angles,np.arctan(y[1]/y[0]))
        
#        print(angles)
        return (np.mean(angles))
            
    def find_dx_dy(self,corners):
        
#        angle=self.find_angle(self.data)
        angle=self.find_angle2()
        print(angle)
        differences=np.array([0,0])
        
        for i in range(self.data.shape[0]//4):
            for j in range(4):
                differences=np.vstack((differences,corners[j]-self.data[i*4+j]))
        
        differences=np.delete(differences,0,0)
#        print(differences)
        
        rotated_differences=np.apply_along_axis(rotate_values,1,differences,angle)
        
        rotated_diff=rotated_differences.T
#        print(rotated_diff)
        self.rotated_diff=abs(rotated_diff)
#        print(self.rotated_diff)
#        return rotated_diff
    
    
    
    def plot_histograms(self):
        ideal_x=0.470
        ideal_y=0.095
        title_label="Mark G "
#        stats=self.get_statistics()
        
        
        plt.figure()
        plt.hist(self.rotated_diff[0],ec='black',color='skyblue')
        plt.title("$\mu = ${0:.4f} $\sigma = ${1:.4f}".format(self.stats[0][1],self.stats[0][2]))
        plt.axvline(ideal_x, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(ideal_x+0.02, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(ideal_x-0.02, color='r', linestyle='dashed', linewidth=1)

        plt.suptitle("{}x values".format(title_label), fontweight='bold')
        plt.show()
        
        plt.figure()
        plt.hist(self.rotated_diff[1],ec='black',color='skyblue')
        plt.suptitle("{}y values".format(title_label), fontweight='bold')
        plt.axvline(ideal_y, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(ideal_y+0.02, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(ideal_y-0.02, color='r', linestyle='dashed', linewidth=1)
        plt.title("$\mu = ${0:.4f} $\sigma = ${1:.4f}".format(self.stats[1][1],self.stats[1][2]))
        plt.show()

    def get_statistics(self):
        mean_values=np.mean(self.rotated_diff,axis=1)
        std_values=np.std(self.rotated_diff,axis=1)
        sem_values=stats.sem(self.rotated_diff,axis=1)
        """
        Ideal values for dx:
            F: 0.1275
            E: 0.250
            I: 0.470
            G: 1.145
        """
        t_vals_x=stats.ttest_1samp(self.rotated_diff[0],0.250)
        """
        Ideal values for dy:
            F: 0.1175
            E: 0.095
            I: 0.095
            G: 0.095
        """
        t_vals_y=stats.ttest_1samp(self.rotated_diff[1],0.095)
        n_vals=self.rotated_diff.shape[1]
        n=np.array([n_vals,n_vals])
        
        
        stat_vals=np.vstack((n,mean_values,std_values,sem_values,np.array([t_vals_x[1],t_vals_y[1]]))).T
        self.stats=stat_vals
        df=pd.DataFrame(stat_vals)

        df.rename(columns={0:'n',1:'mean',2:'std',3:'sem',4:'t pvalue'},index={0:'x',1:'y'},inplace=True)
        print(df)
        print(t_vals_x)
        print(t_vals_y)
        return df

if __name__ == '__main__':
    ph=point_histogram("6-13-PatternMatchPosition-I.csv")
    
    data=pd.read_csv("6-13-Coords.csv", sep=',',header=None)#read csv
    data=np.array(data)#Turn into numpy array
    data=np.delete(data,2,1)#delete z coordinate column - not needed and saves computation time
    
    ph.find_dx_dy(data)
    ph.get_statistics()
    ph.plot_histograms()
    
    corner_dist=np.zeros([4,4])
    for i in range(4):
        for j in range(i+1,4):
            corner_dist[i][j]=corner_dist[j][i]=np.linalg.norm(data[i]-data[j])
    print("\n~~~Corner data~~~")
    print(corner_dist)

    
    