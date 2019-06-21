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
        self.angle=angle
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
        title_label="Mark I "
#        stats=self.get_statistics()
        
        
        plt.figure()
        
        binw=0.0025
        
        binx=np.arange(ideal_x-(binw/2.+0.02),ideal_x+(binw/2.+0.02),binw)
        myhist=plt.hist(self.rotated_diff[0],ec='black',color='skyblue',bins=binx)
        plt.title("$\mu = ${0:.4f} (mm) $\sigma = ${1:.4f} (mm)".format(self.stats[0][1],self.stats[0][2]))
        plt.axvline(ideal_x, color='k', linestyle='dashed', linewidth=1,label="ideal")
        plt.axvline(ideal_x+0.02, color='r', linestyle='dashed', linewidth=1,label="tolerance")
        plt.axvline(ideal_x-0.02, color='r', linestyle='dashed', linewidth=1)
        
        binx2=[i+binw/2. for i in binx] #np.arange(ideal_x-0.02,ideal_x+0.02,0.005)
        plt.xticks(binx2,rotation=45)
        
        for label in myhist.ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        plt.legend()
        plt.xlabel("$\Delta$x Value (mm)")
        plt.ylabel("Counts")

        plt.suptitle("{}$\Delta$x values".format(title_label), fontweight='bold')
        plt.show()
        
        biny=np.arange(ideal_y-(binw/2.+0.02),ideal_y+(binw/2.+0.02),binw)
        
        plt.figure()
        plt.hist(self.rotated_diff[1],ec='black',color='skyblue',bins=biny)
        plt.suptitle("{}$\Delta$y values".format(title_label), fontweight='bold')
        plt.axvline(ideal_y, color='k', linestyle='dashed', linewidth=1,label='ideal')
        plt.axvline(ideal_y+0.02, color='r', linestyle='dashed', linewidth=1,label='tolerance')
        plt.axvline(ideal_y-0.02, color='r', linestyle='dashed', linewidth=1)
        plt.title("$\mu = ${0:.4f} (mm) $\sigma = ${1:.4f} (mm)".format(self.stats[1][1],self.stats[1][2]))
        plt.legend()
        
        biny=[i+binw/2. for i in biny]
        plt.xticks(biny,rotation=45)
        
        plt.xlabel("$\Delta$y Value (mm)")
        plt.ylabel("Counts")
        plt.show()
        
    def plot_fid_to_fid_histo(self):
        
        ideals={'F':[97.366,97.7155],'E':[97.121,97.7605],'G':[95.331,97.7605],'I':[96.681,97.7605]}
        
        mark='I'
        
        binw=0.0025

        binh=np.arange(ideals[mark][0]-(0.01+binw/2),ideals[mark][0]+(0.02+binw/2),binw)
        
        
        horizontal_dist=np.array([])
        vertical_dist=np.array([])
        
        for i in range(self.data.shape[0]//4):

            horizontal_dist=np.append(horizontal_dist,np.linalg.norm(self.data[4*i]-self.data[4*i+3]))
            horizontal_dist=np.append(horizontal_dist,np.linalg.norm(self.data[4*i+1]-self.data[4*i+2]))
            
            vertical_dist=np.append(vertical_dist,np.linalg.norm(self.data[4*i]-self.data[4*i+1]))
            vertical_dist=np.append(vertical_dist,np.linalg.norm(self.data[4*i+2]-self.data[4*i+3]))

        
        plt.figure()
        plt.hist(horizontal_dist,ec='black',bins=binh)
        plt.suptitle("Mark {} Horizontal Distances".format(mark), fontweight='bold')
        plt.title("$\mu = ${0:.4f}(mm); $\sigma = ${1:.4f}(mm); bin size = {2} mm".format(np.mean(horizontal_dist),np.std(horizontal_dist),binw))
        plt.axvline(ideals[mark][0], color='k', linestyle='dashed', linewidth=1,label='ideal')
        plt.ylabel("Count")
        plt.legend()
        binx2=[i+binw/2. for i in binh]
        plt.xticks(binx2,rotation=45)
        
        plt.show()
        
        binv=np.arange(ideals[mark][1]-(0.01+binw/2),ideals[mark][1]+(0.02+binw/2),binw)
        
        
        
        plt.figure()
        plt.hist(vertical_dist,ec='black',bins=binv)
        plt.suptitle("Mark {} Vertical Distances".format(mark), fontweight='bold')
        plt.title("$\mu = ${0:.4f}(mm); $\sigma = ${1:.4f}(mm); bin size = {2} mm".format(np.mean(vertical_dist),np.std(vertical_dist),binw))
        plt.axvline(ideals[mark][1], color='k', linestyle='dashed', linewidth=1,label='ideal')
        plt.ylabel("Count")
        plt.legend()
        binx2=[i+binw/2. for i in binv]
        
        plt.xticks(binx2,rotation=45)
        plt.show()
        
    def show_fid_to_fid_over_time(self):
        
        all_vals=np.array([])

        vals=np.array([])
        
        for i in range(self.data.shape[0]//4):
            all_vals=np.append(all_vals,np.linalg.norm(self.data[0]-self.data[4*i+3]))
        vals=np.array([])    
        for i in range(self.data.shape[0]//4):
            vals=np.append(vals,np.linalg.norm(self.data[2]-self.data[4*i+1]))
        all_vals=np.vstack((all_vals,vals))
        vals=np.array([])
        
        for i in range(self.data.shape[0]//4):
            vals=np.append(vals,np.linalg.norm(self.data[0]-self.data[4*i+1]))
        all_vals=np.vstack((all_vals,vals))
        vals=np.array([])
        for i in range(self.data.shape[0]//4):
            vals=np.append(vals,np.linalg.norm(self.data[2]-self.data[4*i+3]))
        all_vals=np.vstack((all_vals,vals))
        
        
        vals=np.array([])
        
        for i in range(self.data.shape[0]//4):
            vals=np.append(vals,np.linalg.norm(self.data[3]-self.data[4*i]))
        all_vals=np.vstack((all_vals,vals))
        vals=np.array([])    
        for i in range(self.data.shape[0]//4):
            vals=np.append(vals,np.linalg.norm(self.data[1]-self.data[4*i+2]))
        all_vals=np.vstack((all_vals,vals))
        vals=np.array([])
        
        for i in range(self.data.shape[0]//4):
            vals=np.append(vals,np.linalg.norm(self.data[1]-self.data[4*i]))
        all_vals=np.vstack((all_vals,vals))
        vals=np.array([])
        for i in range(self.data.shape[0]//4):
            vals=np.append(vals,np.linalg.norm(self.data[3]-self.data[4*i+2]))
        all_vals=np.vstack((all_vals,vals))
  

            
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2,figsize=[10,10])
#        plt.suptitle("C and A held constant",y=0.93,va='center')
        ax1.plot(all_vals[0],'k-o')
        ax1.title.set_text("D to C, Constant C")
        ax2.plot(all_vals[1],'k-o')
        ax2.title.set_text("A to B, Constant A")
        ax3.plot(all_vals[2],'k-o')
        ax3.title.set_text("C to B, Constant C")
        ax4.plot(all_vals[3],'k-o')
        ax4.title.set_text("D to A, Constant A")
        
        ax5.plot(all_vals[4],'k-o')
        ax5.title.set_text("D to C, Constant D")
        ax6.plot(all_vals[5],'k-o')
        ax6.title.set_text("A to B, Constant B")
        ax7.plot(all_vals[6],'k-o')
        ax7.title.set_text("C to B, Constant B")
        ax8.plot(all_vals[7],'k-o')
        ax8.title.set_text("D to A, Constant D")
        plt.tight_layout()

        

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
#        print(df)
#        print(t_vals_x)
#        print(t_vals_y)
        return df

if __name__ == '__main__':
    ph=point_histogram("6-13-PatternMatchPosition-I.csv")
    
    data=pd.read_csv("6-13-Coords.csv", sep=',',header=None)#read csv
    data=np.array(data)#Turn into numpy array
    data=np.delete(data,2,1)#delete z coordinate column - not needed and saves computation time
    
    ph.find_dx_dy(data)
    ph.get_statistics()
#    ph.plot_histograms()
    ph.plot_fid_to_fid_histo()
#    ph.show_fid_to_fid_over_time()
    
    """
    corner_dist=np.zeros([4,4])
    for i in range(4):
        for j in range(i+1,4):
            corner_dist[i][j]=corner_dist[j][i]=np.linalg.norm(data[i]-data[j])
    print("\n~~~Corner data~~~")
    print(corner_dist)
    """
    

    
    