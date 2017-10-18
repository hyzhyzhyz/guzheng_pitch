import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def contour_truth(debug=False):
    bin_file='./data/test_bin.pv'
    fre_file='./data/test_fre.pv'
    bin_truth=[]
    with open(bin_file,'r') as f:
        for line in f:
            bin_truth.append(map(float,line.split()))
    f.close()
    fre_truth=[]
    with open(fre_file,'r') as f:
        for line in f:
            fre_truth.append(map(float,line.split()))
    f.close()
    #print fre_truth[0]
    contour_bin=[]
    contour_fre=[]
    for index,bin_truth_ite in enumerate(bin_truth):
        for i_iter in range(1,len(bin_truth_ite)):
            if bin_truth_ite[i_iter]>0:
                temp=[bin_truth_ite[0],bin_truth_ite[0],bin_truth_ite[i_iter]]
                start=index+1
                while(start<len(bin_truth) and bin_truth_ite[i_iter] in bin_truth[start]):
                    index_temp=bin_truth[start].index(bin_truth_ite[i_iter])
                    bin_truth[start][index_temp]=-1
                    start+=1
                bin_truth_ite[i_iter]=-1
                temp[1]=bin_truth[start-1][0]
                contour_bin.append(temp)
                temp=[]
    if debug:
        print np.array(contour_bin).shape
    for index,fre_truth_ite in enumerate(fre_truth):
        for i_iter in range(1,len(fre_truth_ite)):
            if fre_truth_ite[i_iter]>0:
                temp=[fre_truth_ite[0],fre_truth_ite[0],fre_truth_ite[i_iter]]
                start=index+1
                while(start<len(fre_truth) and fre_truth_ite[i_iter] in fre_truth[start]):
                    index_temp=fre_truth[start].index(fre_truth_ite[i_iter])
                    fre_truth[start][index_temp]=-1
                    start+=1
                fre_truth_ite[i_iter]=-1
                temp[1]=fre_truth[start-1][0]
                contour_fre.append(temp)
                temp=[]
    if debug:
        print np.array(contour_fre).shape
    return contour_bin,contour_fre
if __name__=='__main__':
   contour_bin,contour_fre= contour_truth(True)
   for contour_bin_iter in contour_bin:
       if contour_bin_iter[0]>1:
           break
       while (contour_bin_iter[0]<=contour_bin_iter[1]):
           plt.scatter(contour_bin_iter[0],contour_bin_iter[2])
           contour_bin_iter[0]+=0.01
   plt.savefig('truth.jpg')
            


            
