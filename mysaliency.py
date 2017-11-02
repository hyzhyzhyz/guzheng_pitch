# -*- coding: utf-8 -*-
"""
Description:
@author: Gordon
"""

import sys, csv, os
from essentia import *
from essentia.standard import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot    as plt
import numpy as np
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
#import stft as STFT
import _savitzky_golay as savfilt
from scipy.signal import medfilt
import math



def My_FrameGenerator(audio, frameSize, hopSize):
    '''
    这是自己写的分帧代码，essentia的分帧代码有点问题
    '''
    frame_all=[]
    frame_num=(len(audio)-frameSize)/hopSize+1
    for i in range(frame_num):
        frame_temp=audio[i*hopSize:i*hopSize+frameSize]
        frame_all.append(frame_temp)
    return np.asarray(frame_all)

def run_pitch_salience_function_shs(peak_frequencies,peak_magnitudes):
    '''
    这是分谐波叠加的函数
    对应HPCP的函数
    '''
    alpha=(2.0**(1.0/24.0)-1)
    s_log=np.zeros(600)
    for fre,mag in zip(peak_frequencies,peak_magnitudes):
        Harmonic=[]
        for i in range(1,11):
            Harmonic.append(fre*i)
        temp_energy=0
        for index,harmonic_iter in enumerate(Harmonic):
            diff=abs(np.array(peak_frequencies)-harmonic_iter)
            min_diff=min(diff)
            if min_diff<alpha*harmonic_iter:
                index_mag=np.where(diff==min_diff)[0]
                index_mag=index_mag.tolist()[0]
                mag_harmonic=peak_magnitudes[index_mag]
                temp_energy+=0.8**(index)*mag_harmonic
        if fre>=55 and fre<=1760  and temp_energy>0.0:
            temp_bin=round(120*math.log(fre/55.0,2))-1
            s_log[int(temp_bin)]=temp_energy
    return essentia.array(s_log)




def musicVocaliNonVocalic(audio, hopSize=128, frameSize=2048, sampleRate=44100, debug=False):
    #filename = '../segwav/3harmonicComp.wav'
    #hopSize = 128
    #frameSize = 2048
    #sampleRate = 44100
    #guessUnvoiced = True
    Salience_peaks=[]
    
    run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
    run_spectrum = Spectrum(size=frameSize * 4)
    run_spectral_peaks = SpectralPeaks(minFrequency=50,
                                   maxFrequency=10000,
                                   maxPeaks=100,
                                   sampleRate=sampleRate,
                                   magnitudeThreshold=0,
                                   orderBy="magnitude")
    run_pitch_salience_function = PitchSalienceFunction(magnitudeThreshold=60)
    run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks(minFrequency=55, maxFrequency=1760)
#    run_pitch_contours = PitchContours(hopSize=hopSize, peakFrameThreshold=0.7)
#    run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=guessUnvoiced,
#                                                hopSize=hopSize)
    pool = Pool();
    #audio = MonoLoader(filename = filename)()
    #audio = EqualLoudness()(audio)
    i=1
    for frame in My_FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
    #for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        #print type(frame)
        frame = run_windowing(frame)
        spectrum = run_spectrum(frame)
        specGram = pool.add('allframe_spectrum', spectrum);
        peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
        salience = run_pitch_salience_function_shs(peak_frequencies, peak_magnitudes)
        #salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
        Salience_fin=pool.add('allframe_salience',salience);
        salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
        if debug and len(salience_peaks_saliences)>0:
            #print salience_peaks_saliences
            salience_max=max(salience_peaks_saliences)
            for fre, sal in zip( salience_peaks_bins, salience_peaks_saliences):
                if sal>salience_max*0.4:
                    plt.scatter(i,fre)

        #print salience_peaks_bins
        salience_peak_temp=np.zeros([600,1])
        for _bin,_peak in zip(salience_peaks_bins,salience_peaks_saliences):
            salience_peak_temp[int(_bin)]=_peak
        if len(Salience_peaks)==0:
            Salience_peaks=salience_peak_temp
        else:
            Salience_peaks=np.column_stack((Salience_peaks,salience_peak_temp))
        salSum = np.sum(np.power(salience_peaks_saliences, 2))
        pool.add('salienceSum', salSum)
        i+=1
    if debug:
        plt.savefig('fre_peak.jpg')
        plt.close()
        #print 'frame_num',FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize).num_frames()
    specGram = pool['allframe_spectrum']
    Salience_fin=pool['allframe_salience']
    totalSalienceEnrg = pool['salienceSum']     
    totalSalienceEnrg = totalSalienceEnrg/np.max(totalSalienceEnrg)
    timeAxis = ((hopSize) * np.arange(np.size(totalSalienceEnrg)))/float(sampleRate)
    audioTime = np.arange(np.size(audio))/float(sampleRate)
    
    return specGram,Salience_peaks
