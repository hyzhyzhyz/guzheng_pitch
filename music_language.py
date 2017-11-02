#encoding:utf-8
import numpy as np
import copy 

def lianzou(contour):
    '''
    花音(连奏)，板前花，定义为
    花是古筝中连续拨动琴弦的一种弹奏技巧,连续拨动至少几根才算做是花音？
    input:contour,提取的轮廓线
    contour的排序是按照最大salience来排序的，并不是按照时间的先后顺序来排序的。
    连奏的个数越多，最后是基频的可能性就越大
    return:
    '''
    contour_lianzou=[False]*len(contour)
    #首先对contour进行排序，按照起始帧来进行排序
    contour_sort=copy.copy(contour)
    #contour_sort=sorted(contour,key=lambda x:x[0][2])
    for index,contour_temp in enumerate(contour_sort):
        #从起始帧开始往后面找
        #找到花音的轮廓线就return True
        start=index+1
        while(start<len(contour_sort) and contour_sort[start][0][2]<contour_temp[-1][2]+5):
            if contour_sort[start][0][2]<contour_temp[-1][2]-5:
                start+=1
                continue
            bin_diff = abs(contour_sort[start][0][1]-contour_temp[-1][1])
            if bin_diff>15 and bin_diff<35 :
                contour_lianzou[start]=True
                contour_lianzou[index]=True
            start+=1
    return contour_lianzou
def chanyin(contour):
    '''
    颤音：在弹奏的时候按住古筝的琴弦而得到的音叫做颤音
    包括滑音，点音等
    颤音有可能会存在于轮廓线内部
    颤音有可能会存在于轮廓线之间？？？
    '''
    contour_chanyin=[False]*len(contour)
    for index,contour_temp in enumerate(contour):
        nums=[]
        for contour_temp_iter in contour_temp:
            nums.append(contour_temp_iter[1])
        contour_chanyin[index]=continue_num(nums)
    return contour_chanyin
def continue_num(nums):
    '''
    给定一个num，寻找最大连续递增的序列长度
    增加的序列长度由快到慢，变化至少10个bin
    如果0.1s变化了至少10个bin,就认为是滑音
    '''
    for i in range(len(nums)-10):
        if abs(nums[i+10]-nums[i])>10:
            return True
    if len(nums)<10:
        return abs(nums[-1]-nums[0])>10
    return False
        
        
