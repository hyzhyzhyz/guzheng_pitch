#encoding:utf-8
import numpy as np
import copy 

def lianzou(contour):
    '''
    花音(连奏)，板前花，定义为
    花是古筝中连续拨动琴弦的一种弹奏技巧,连续拨动至少几根才算做是花音？
    input:contour,提取的轮廓线
    contour的排序是按照最大salience来排序的，并不是按照时间的先后顺序来排序的。
    return:
    '''
    contour_lianzou=[False]*len(contour)
    #首先对contour进行排序，按照起始帧来进行排序
    contour_sort=copy.copy(contour)
    contour_sort=sorted(contour,key=lambda x:x[0][2])
    for index,contour_temp in enumerate(contour_sort):
        #从起始帧开始往后面找
        #找到花音的轮廓线就return True
        start=index+1
        while(start<len(contour_sort) and contour_sort[start][0][2]<contour_temp[0][2]+5):
            if contour_sort[start][0][2]<contour_temp[0][2]-5:
                start+=1
                continue
            bin_diff = abs(contour_sort[start][0][1]-contour_temp[0][1])
            if bin_diff in range(15,35):
                contour_lianzou[index]=True
                contour_lianzou[start]=True
            start+=1

        
        
