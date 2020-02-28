from __future__ import absolute_import, division, print_function
import soundfile as sf
import subprocess
import os
import sys
from Silence_Remove import audioBasicIO
from Silence_Remove import audioSegmentation as aS
from pydub import AudioSegment
import librosa
import numpy as np
import ntpath
from shutil import copy2
import time
import re,string

def normalize(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    s = regex.sub(' ', s)
    s = s.replace(' 0 ',' không ')
    s = s.replace(' 1 ',' một ')
    s = s.replace(' 2 ',' hai ')
    s = s.replace(' 3 ',' ba ')
    s = s.replace(' 4 ',' bốn ')
    s = s.replace(' 5 ',' năm ')
    s = s.replace(' 6 ',' sáu ')
    s = s.replace(' 7 ',' bảy ')
    s = s.replace(' 8 ',' tám ')
    s = s.replace(' 9 ',' chín ')
    s = s.lower()
    s = ' '.join(s.split())
    return s

def absoluteFilePaths(directory):
    paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            paths.append(os.path.abspath(os.path.join(dirpath, f)))
            
    return paths

def convert_to_16k(in_path,out_path):
    y, s = librosa.load(in_path, sr=8000)
    y_16k = librosa.resample(y, s, 16000)
    sf.write(out_path, y_16k, 16000, format='WAV', subtype='PCM_16')

def silenceRemovalWrapper(inputFile, smoothingWindow=0.5, weight=0.2, saveFile=False):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")
    newName = inputFile.split('/')[-1].split('.')[0]
    [fs, x] = audioBasicIO.readAudioFile(inputFile)
    segmentLimits = aS.silenceRemoval(x, fs, 0.03, 0.03,
                                      smoothingWindow, weight, False)
    for i, s in enumerate(segmentLimits):
        strOut = "{0:s}_{1:.2f}-{2:.2f}.wav".format(inputFile[0:-4], s[0], s[1])
        if saveFile:
            wavfile.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])
    return segmentLimits

def chunk_audio(file_path, output_path, max_len = 12):
    
    file_name = ntpath.basename(file_path)
    audio = AudioSegment.from_wav(file_path)
    if len(audio) / 1000  > max_len:
        segs = silenceRemovalWrapper(file_path)
        segs = sorted([j for i in segs for j in i])
        points = []
        total = 0
        
        for i in range(1,len(segs)):
            gap = segs[i] - segs[i-1]
            if total + gap > max_len:
                points.append(segs[i-1])
                total = gap
                while(total > max_len):
                    points.append(points[-1] + max_len)
                    total -= max_len
            else:
                total += gap

        points.append(segs[-1])
        if points[0] != segs[0]:
            points.insert(0,segs[0])
        points = [int(p*1000) for p in points]
        
        for i in range(1,len(points)):
            part = audio[points[i-1]:points[i]]
            path_to_write = os.path.join(output_path,file_name.replace('.wav','_' + str(i-1) + '.wav'))
            part.export(path_to_write, format='wav')
    else:
        path_to_write = os.path.join(output_path, file_name.replace('.wav','_0.wav'))
        copy2(file_path,path_to_write)
    
    if output_path in file_path:
        os.remove(file_path)

def read_result(path):
    trans = {}

    with open(path) as f:
        data = f.read().split('\n')
        data = data[:-1]

    for d in data:
        end = d.find('(')
        pred = d[0:end-1]
        file_name = d[end:].replace('(','').replace(')','')
        index = int(file_name.split('_')[-1].replace('.wav',''))
        base_name = int(file_name.split('_')[0])

        if base_name not in trans:
            trans[base_name] = [(index,pred)]
        else:
            trans[base_name].append((index,pred))

    for name in trans:
        trans[name] = sorted(trans[name], key = lambda x: x[0])
        trans[name] = ' '.join([t[1] for t in trans[name]])

    return trans