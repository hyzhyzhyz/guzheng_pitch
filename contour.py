#encoding:utf-8
from essentia import*
from essentia.standard import*
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate(salience,allowed_pitch_distance,allowed_piths,max_ratio):
    '''
    allowed_pitch_distance:构造轮廓线时允许的最大bin差
    allowed_pitchs:一帧音频最多允许的轮廓线条数
    max_ratio:表示当小于该帧最大值的多少倍时，将不再进行轮廓线构建
    '''
    contour=[]
    max_frame=[]##表示每一帧中salience的最大值
    salience=np.mat(salience)
    #print salience[:,16]
    raw,column=salience.shape
    for i in range(column):
        max_frame.append(salience[:,i].max())
    #print max_frame
    #print max_frame[0]*max_ratio
    _position=np.argmax(salience)
    _bin,_frame=divmod(_position,column)
    max_salience=salience[_bin,_frame]
    salience[_bin,_frame]=0
    flag_ones=np.ones(column)*allowed_piths
    #print _frame
    while max_salience>0:
        contour_forward=[]
        contour_backward=[]
        if max_salience<max_frame[_frame]:
            _position=np.argmax(salience)
            _bin,_frame=divmod(_position,column)
            max_salience=salience[_bin,_frame]
            salience[_bin,_frame]=0
        first_temp=[max_salience,_bin,_frame]
        contour_forward.append(first_temp)
        contour_point_num=0
        while True:
            frame_iter=contour_forward[contour_point_num][2]+1
            if(frame_iter>column-1):
                break
            findnextpoint=0
            for bin_range in range(allowed_pitch_distance):
                Binid_temp=0
                Frameid_temp=0
                #print frame_iter,max_frame[frame_iter],max_frame[frame_iter]*max_ratio
                if(salience[min(599,_bin+bin_range),frame_iter]>max_frame[frame_iter]*max_ratio):
                    #print salience[min(599,_bin+bin_range),frame_iter]
                    Binid_temp=min(599,_bin+bin_range)
                    Frameid_temp=frame_iter
                    findnextpoint=1
                else:
                    salience[min(599,_bin+bin_range),frame_iter]=0
                if(salience[max(0,_bin-bin_range),frame_iter]>max_frame[frame_iter]*max_ratio):
                    #print salience[max(0,_bin-bin_range),frame_iter]
                    if findnextpoint and salience[max(0,_bin-bin_range),frame_iter]>salience[Binid_temp,frame_iter]:
                        Binid_temp=max(0,_bin-bin_range)
                        Frameid_temp=frame_iter
                        findnextpoint=1
                    if findnextpoint==0:
                        Binid_temp=max(0,_bin-bin_range)
                        Frameid_temp=frame_iter
                        findnextpoint=1
                else:
                    salience[max(0,_bin-bin_range),frame_iter]=0
                if(findnextpoint):
                    _bin=Binid_temp
                    _frame=frame_iter
                    break
            if findnextpoint:
                first_temp=[salience[_bin,_frame],_bin,_frame]
                #print salience[_bin,_frame],max_frame[frame_iter]*max_ratio
                #print first_temp
                contour_forward.append(first_temp)
                contour_point_num+=1
                salience[_bin,_frame]=0
            else:
                break
        #print 'contour_forward',contour_forward
        contour_backward.append(contour_forward[0])
        _bin=contour_forward[0][1]
        _frame=contour_forward[0][2]
        contour_point_num=0
        while True:
            #print contour_backward[0]
            frame_iter=contour_backward[contour_point_num][2]-1
            if(frame_iter<0):
                break
            findnextpoint=0
            for bin_range in range(allowed_pitch_distance):
                Binid_temp=0
                Frameid_temp=0
                if(salience[min(599,_bin+bin_range),frame_iter]>max_frame[frame_iter]*max_ratio):
                    Binid_temp=min(599,_bin+bin_range)
                    Frameid_temp=frame_iter
                    findnextpoint=1
                if(salience[max(0,_bin-bin_range),frame_iter]>max_frame[frame_iter]*max_ratio):
                    if findnextpoint and salience[max(0,_bin-bin_range),frame_iter]>salience[Binid_temp,frame_iter]:
                        Binid_temp=max(0,_bin-bin_range)
                        Frameid_temp=frame_iter
                        findnextpoint=1
                    if findnextpoint==0:
                        Binid_temp=max(0,_bin-bin_range)
                        Frameid_temp=frame_iter
                        findnextpoint=1
                if(findnextpoint):
                    _bin=Binid_temp
                    _frame=frame_iter
                    break
            if findnextpoint:
                first_temp=[salience[_bin,_frame],_bin,_frame]
                #print salience[_bin,_frame],max_frame[_frame]
                contour_backward.append(first_temp)
                contour_point_num+=1
                salience[_bin,_frame]=0
            else:
                break
        #将contour_backward和contour_forward合在一起
        contour_backward_temp=[]
        for i in range(len(contour_backward)):
            contour_backward_temp.append(contour_backward[len(contour_backward)-1-i])
        for i in range(1,len(contour_forward)):
            contour_backward_temp.append(contour_forward[i])
        #print contour_backward_temp
        if (len(contour_backward_temp)>2):
            contour.append(contour_backward_temp)
            #print contour_backward_temp
        _position=np.argmax(salience)
        _bin,_frame=divmod(_position,column)
        max_salience=salience[_bin,_frame]
        salience[_bin,_frame]=0




    return contour
def character(all_contour,salience):
    '''
    character函数表述的是生成轮廓线的特征
    轮廓线特征有:
    轮廓线的bin均值
    bin方差
    能量均值
    能量方差
    轮廓线的长度
    轮廓线的总能量
    轮廓线的归一化平均能量（轮廓线的归一化能量是指每一帧的能量除以最大值能量之后的总和）
    轮廓线的后续长度（因为古筝弹奏时有余音的存在，所以存在余音的更有可能是古筝基频）
    '''
    max_frame=[]##表示每一帧中salience的最大值
    max_ratio=0.1##此处的max_ratio并不是为了轮廓线构建而用的，而是用来寻找后续长度
    salience=np.mat(salience)
    #print salience[:,16]
    raw,column=salience.shape
    for i in range(column):
        max_frame.append(salience[:,i].max())
    #print max_frame
    #print max_frame[0]*max_ratio
    #_position=np.argmax(salience)
    #_bin,_frame=divmod(_position,column)
    contour_character=[]
    for contour in all_contour:
        contour_character_temp=[]
        contour_character_temp.append(np.mean(np.array(contour).T[1]))
        contour_character_temp.append(np.std(np.array(contour).T[1]))
        contour_character_temp.append(np.mean(np.array(contour).T[0]))
        contour_character_temp.append(np.std(np.array(contour).T[0]))
        contour_character_temp.append(len(contour))
        contour_character_temp.append(np.sum(np.array(contour).T[0]))
        mean_salience=0.0
        for contour_frame in contour:
            mean_salience+=contour_frame[0]/max_frame[contour_frame[2]]
        contour_character_temp.append(mean_salience/len(contour))
        len_continue=0
        frame_iter=contour[-1][2]
        _bin=contour[-1][1]
        while True:
            frame_iter+=1
            if frame_iter>=column:
                break
            findnextpoint=0
            for bin_range in range(7):
                Binid_temp=0
                Frameid_temp=0
                #print frame_iter,max_frame[frame_iter],max_frame[frame_iter]*max_ratio
                if(salience[min(599,_bin+bin_range),frame_iter]>max_frame[frame_iter]*max_ratio):
                    #print salience[min(599,_bin+bin_range),frame_iter]
                    Binid_temp=min(599,_bin+bin_range)
                    Frameid_temp=frame_iter
                    findnextpoint=1
                else:
                    salience[min(599,_bin+bin_range),frame_iter]=0
                if(salience[max(0,_bin-bin_range),frame_iter]>max_frame[frame_iter]*max_ratio):
                    #print salience[max(0,_bin-bin_range),frame_iter]
                    if findnextpoint and salience[max(0,_bin-bin_range),frame_iter]>salience[Binid_temp,frame_iter]:
                        Binid_temp=max(0,_bin-bin_range)
                        Frameid_temp=frame_iter
                        findnextpoint=1
                    if findnextpoint==0:
                        Binid_temp=max(0,_bin-bin_range)
                        Frameid_temp=frame_iter
                        findnextpoint=1
                else:
                    salience[max(0,_bin-bin_range),frame_iter]=0
                if(findnextpoint):
                    _bin=Binid_temp
                    _frame=frame_iter
                    break
            if findnextpoint:
                len_continue+=1
            else:
                break
        contour_character_temp.append(len_continue)
        contour_character.append(contour_character_temp)

        #for contour_frame in contour:
            #print contour_frame
    return contour_character
def filter(contour_all,contour_character,contour_lianzou,contour_chanyin):
    '''
    input:
    contour_all:所有的轮廓线
    contour_character:提取的轮廓线特征
    contour_lianzou：轮廓线是否存在连奏属性
    contour_chanyin：轮廓线是否存在颤音属性
    output:
    contour_filter_all:滤除不要的轮廓线留下的剩下的轮廓线
    '''
    
