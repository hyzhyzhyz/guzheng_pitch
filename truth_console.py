#encoding: utf-8
import numpy as np
import os
import sys
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def contour_truth(file_name, debug=False):
    print file_name
    bin_file = file_name
    #bin_file='../label_data/bin/1.txt'
    #fre_file='../label_data/fre/'
    bin_truth=[]
    with open(bin_file,'r') as f:
        for line in f:
            bin_truth.append(map(float,line.split()))
    f.close()
    '''
    fre_truth=[]
    with open(fre_file,'r') as f:
        for line in f:
            fre_truth.append(map(float,line.split()))
    f.close()
    '''
    #print fre_truth[0]
    contour_bin=[]
    contour_fre=[]
    for index,bin_truth_ite in enumerate(bin_truth):
        for i_iter in range(1,len(bin_truth_ite)):
            if bin_truth_ite[i_iter]>0:
                temp=[]
                temp.append([1 ,bin_truth_ite[i_iter],  int(bin_truth_ite[0]*100)-1])#因为标注的数据从1开始，转化为从0开始
                start=index+1
                bin_generater=bin_truth_ite[i_iter]
                bin_truth_ite[i_iter] = -1
                while(start<len(bin_truth)):
                    iscontinue = False
                    for start_iter in range(1,len(bin_truth[start])):
                        if bin_generater>bin_truth[start][start_iter]-10 and bin_generater<bin_truth[start][start_iter]+10:
                            iscontinue=True
                            temp.append([1,  bin_truth[start][start_iter],  int(bin_truth[start][0]*100)-1])
                            bin_generater=bin_truth[start][start_iter]
                            bin_truth[start][start_iter]=-1
                            break
                    if not iscontinue:
                        break
                    start+=1
                contour_bin.append(temp)
    if debug:
        print len(contour_bin)
        #print contour_bin[0:10]
        '''
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
        '''
    #print len(bin_truth)
    return contour_bin,contour_fre,len(bin_truth)
if __name__=='__main__':
   contour_bin,contour_fre, frame_all = contour_truth('../label_data/bin/1.txt', True)
   
   '''
   contour_bin_name='./data/contour_bin.h5py'
   f1=h5py.File(contour_bin_name,'w')
   f1.create_dataset('data',data=contour_bin)

   contour_fre_name='./data/contour_fre.h5py'
   f2=h5py.File(contour_fre_name,'w')
   f2.create_dataset('data', data=contour_fre)
   '''
   for bin_iter in contour_bin:
       for bin_iter_iter in bin_iter:
           plt.scatter(bin_iter_iter[2],bin_iter_iter[1])
   plt.savefig('truth.jpg')
            


            
