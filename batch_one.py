from essentia import *
from essentia.standard import *
import numpy as np
import essentiaSpecGram as Spec
import saliencyBasedVUV as vuv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wave
import contour
import os
import copy

def batch_one(inputFile,debug=False):
    #inputFile='./cuo.wav'
    if debug:
        print inputFile
    wavefile=wave.open(inputFile,'r')
    fs=wavefile.getframerate()
    audio=MonoLoader(filename=inputFile)()
    #spectrogram=Spec.essentiaSpect(audio,fs/100,2048,fs)
    spectrogram,salience,vocalBeg,vocalEnd,totalSalienceEnrg=vuv.musicVocaliNonVocalic(audio,fs/100,2048,fs)
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
    contour_all=contour.generate(copy.copy(salience),6,6,0.7)
    for i in range(len(contour_all)):
        for j in range(len(contour_all[i])):
            plt.scatter(contour_all[i][j][2],contour_all[i][j][1])
    plt.xlim((0,salience.shape[1]))
    plt.ylim((0,salience.shape[0]))
    my_x_ticks=np.arange(0,salience.shape[1],1)
    my_y_ticks=np.arange(0,salience.shape[0],float(salience.shape[0])/salience.shape[1])
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.savefig('./picture/'+file_name+'_contour.jpg')
    plt.close()
    contour_character=contour.character(copy.copy(contour_all),copy.copy(salience))
    #print contour_all[0]
if __name__=='__main__':
    batch_one('./cuo.wav',True)
