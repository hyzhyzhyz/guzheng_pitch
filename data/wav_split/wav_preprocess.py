import essentia
from essentia.standard import *
import numpy as np

loader=AudioLoader(filename='../test_16000.wav')
audio=loader()[0]
print len(audio)
fs=loader()[1]
print fs
channels=loader()[2]
print channels
writer = AudioWriter(_audio=audio, filename = '../test_test.wav', format = 'wav', sampleRate=16000)
writer()
