#encoding:utf-8
import batch_one
import os

if __name__=='__main__':
    wav_path='./data/wav'
    files=[os.path.join(wav_path,f) for f in os.listdir(wav_path)
            if os.path.isfile(os.path.join(wav_path,f))]
    for f in files:
        batch_one.batch_one(f,True)
    #print files
