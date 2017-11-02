#encoding:utf-8
import numpy as np
import h5py
import batch_one_copy
import truth_console
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import truth_console

def contour_evaluate(contour_truth, contour_test):
    '''
    input:
    contour_truth:ground_truth轮廓线,数据结构是[[1,bin,frame][]...],由于在label里面没有salience，salience统一为1
    contour_test:生成的提取轮廓线,数据结构是[[salience,bin,frame]...][[[0.01,340,1],[]],[],[]]
    轮廓线为真的定义是:contour_test在contour_truth中找到大于一半的部分在contour_truth,即算轮廓线存在
    output:
    recall:召回率
    accuracy:准确率
    contour_flag:contour_test中轮廓线为真就是true，否则就是false
    定义超过60%的轮廓线重合即认为轮廓线召回
    '''
    N=0#N表示contour_test里面的正确轮廓线的个数
    M=0#M表示在contour_trth里面被召回的轮廓线个数
    F=0#F表示在contour_all里面，没有在contour_truth里面
    truth_len=len(contour_truth)
    test_len=len(contour_test)
    contour_flag=[False]*test_len
    contour_truth_flag=[False]*truth_len
    #contour_truth=np.array(contour_truth)
    #contour_test = np.array(contour_test)
    for index,contour_iter in enumerate(contour_test):
        #print type(contour_iter)
        contour_iter=np.array(contour_iter)
        mean_bin=np.mean(contour_iter[:,1])
        start_frame=contour_iter[0,2]
        end_frame=contour_iter[-1,2]
        #print start_frame
        for index_truth, contour_truth_iter in enumerate(contour_truth):
            contour_truth_iter=np.array(contour_truth_iter)
            if contour_truth_iter[0,2]>end_frame:
                break
            if contour_truth_iter[-1,2]<start_frame:
                continue
            same=0
            start_test=0
            start_truth=0
            #首先找到重叠起始点
            while True:
                if contour_iter[start_test,2]>=contour_truth_iter[0,2] and \
                contour_iter[start_test,2]<=contour_truth_iter[-1,2]:
                    start_truth=int(contour_iter[start_test,2]-contour_truth_iter[0,2])
                    break
                else:
                    start_test+=1
            while start_test<len(contour_iter) and start_truth<len(contour_truth_iter):
                #print start_truth
                #print contour_truth_iter[start_truth,1]
                if contour_iter[start_test,1]>contour_truth_iter[start_truth,1]-10 and \
                contour_iter[start_test,1]<contour_truth_iter[start_truth,1]+10:
                    same+=1
                    start_test+=1
                    start_truth+=1
                else:
                    break
            if same>=len(contour_iter)*0.1:
                contour_flag[index]=True
                contour_truth_flag[index_truth]=True
                N+=1
    M=sum(contour_truth_flag)
    return  contour_flag, float(M)/float(truth_len), float(N)/float(test_len) 
def frame_evaluate(contour_truth,contour_test,frame_all):
    '''
    input:
    frame_all:总共有多少帧，注意提取出的轮廓线和label的帧数会不一样，取较大的一个，
    提取出的轮廓线和lable可能会相差1
    contour_truth:ground_truth轮廓线
    contour_all:生成的真实轮廓线
    frame_all:总共的帧数
    output:
    recall:召回率
    accuracy:准确率
    contour_flag:contour_all中轮廓线为真就是true，否则就是false
    '''
    frame_truth = [[]]*frame_all
    frame_test = [[]]*frame_all
    for contour_truth_iter in contour_truth:
        for truth_point in contour_truth_iter:
            if truth_point[2]<frame_all:
                frame_truth[truth_point[2]].append(truth_point[1])
    for contour_test_iter in contour_test:
        for test_point in contour_test_iter:
            if test_point[2]<frame_all:
                frame_test[test_point[2]].append(test_point[1])
    sum_truth = 0#在label中的所有基频点
    sum_test = 0#在提取轮廓线中的所有基频点
    sum_acc = 0#在label中也在轮廓线中的基频点
    for frame_iter in range(frame_all):
        sum_truth += len(frame_truth[frame_iter])
        sum_test += len(frame_test[frame_iter])
        for label in frame_truth[frame_iter]:
            for pitch in frame_test[frame_iter]:
                if abs(pitch-label)<=8:
                    sum_acc+=1
                    break
    return float(sum_acc)/float(sum_truth), float(sum_acc)/float(sum_test) 
if __name__=='__main__':
    contour_truth , contour_fre, frame_all_truth  = truth_console.contour_truth(file_name='../label_data/bin/2.txt')
    #print contour_truth
    contour_test , frame_all_test = batch_one_copy.batch_one(inputFile = '../label_data/wav/2.wav',debug=True)
    frame_all = min(frame_all_truth, frame_all_test)
    print frame_all
    contour_flag , recall , accuracy = contour_evaluate(np.array(contour_truth), contour_test)
    for i in range(len(contour_flag)):
        if contour_flag[i]:
            for j in contour_test[i]:
                plt.scatter(j[2],j[1], c='red')
        else:
            for j in contour_test[i]:
                plt.scatter(j[2],j[1], c='green')
    plt.savefig('recall.jpg')
    plt.close()

    for contour_bin_iter in contour_truth:
        for contour_truth_point in contour_bin_iter:
            plt.scatter(contour_truth_point[2],contour_truth_point[1])
    plt.savefig('recall_truth.jpg')
    plt.close()
    print ('recall_contour',recall)
    print ('accuracy_contour', accuracy)
    recall_frame, accuracy_frame = frame_evaluate(contour_truth, contour_test, frame_all)

    print ('recall_frame', recall_frame)
    print ('accuracy_frame', accuracy_frame)


