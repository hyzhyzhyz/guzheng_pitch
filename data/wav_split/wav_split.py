import wave
import h5py
import numpy as np
from essentia.standard import *


wavefile=wave.open('../test_16000.wav','r')
f1=h5py.File('../contour_bin.h5py','r')
contour_bin=np.asarray(f1['data'])
f2=h5py.File('../contour_fre.h5py','r')
contour_fre=np.asarray(f2['data'])
fs=wavefile.getframerate()
points = wavefile.getnframes()
audio = wavefile.readframes(points)
#audio=float(audio)
print audio[0]
print type(audio)
loader = MonoLoader(filename='../test1.wav')
audio = loader()

print type(audio)
#print MonoLoader.__file__
bin_file='../test_bin.pv'
fre_file='../test_fre.pv'
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


print fs,len(audio),len(bin_truth)
    
