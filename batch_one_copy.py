#encoding: utf-8
from essentia import *
from essentia.standard import *
import numpy as np
import essentiaSpecGram as Spec
import mysaliency as vuv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wave
import contour
import os
import copy
import music_language

def batch_one(inputFile,debug=False):
    #inputFile='./cuo.wav'
    if debug:
        print inputFile
    wavefile=wave.open(inputFile,'r')
    fs=wavefile.getframerate()
    audio=MonoLoader(filename=inputFile, sampleRate = fs)()
    if debug:
        print ('audio',len(audio))
    #spectrogram=Spec.essentiaSpect(audio,fs/100,2048,fs)
    spectrogram,salience = vuv.musicVocaliNonVocalic(audio,fs/100,2048,fs,True)
    spectrogram=np.transpose(spectrogram)
    if debug:
        print spectrogram.shape[0]
        print salience.shape
    #print salience
    file_name=os.path.basename(os.path.splitext(inputFile)[0] )  

    fig=plt.figure(figsize=(15,20))
    ax1=fig.add_subplot(111)
    ax1.imshow(salience,cmap=plt.cm.gray)
    fig.gca().invert_yaxis()
    #plt.axis([0,spectrogram.shape[1],0,199*fs/spectrogram.shape[0]])
    #fig.show()
    fig.savefig('./picture/'+file_name+'_salience.jpg')
    plt.close()
    contour_all=contour.generate(copy.copy(salience),8,6,0.4)
    #轮廓线排序按照起始帧排序
    contour_all=sorted(contour_all,key=lambda x:x[0][2])
    contour_lianzou=music_language.lianzou(copy.copy(contour_all))
    contour_chanyin=music_language.chanyin(copy.copy(contour_all))
    for i in range(len(contour_all)):
        for j in range(len(contour_all[i])):
            if contour_lianzou[i]:
                plt.scatter(contour_all[i][j][2],contour_all[i][j][1],c='red')
            else:
                plt.scatter(contour_all[i][j][2],contour_all[i][j][1],c='green')
    '''
    plt.xlim((0,salience.shape[1]))
    plt.ylim((0,salience.shape[0]))
    my_x_ticks=np.arange(0,salience.shape[1],salience.shape[1]/20)
    my_y_ticks=np.arange(0,salience.shape[0],20*float(salience.shape[0])/salience.shape[1])
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    '''
    plt.savefig('./picture/'+file_name+'_contour.jpg')
    plt.close()
    contour_character=contour.character(copy.copy(contour_all),copy.copy(salience))
    return contour_all,salience.shape[1]
    #print contour_all[0]
if __name__=='__main__':
    batch_one('../label_data/wav/1.wav',True)
