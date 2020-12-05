from wav2vec import EmbeddingDatasetWriter
from wav2vec import Prediction
import argparse
import os
from pydub import AudioSegment
import ntpath
from os.path import abspath
import time
from multiprocessing import Pool
import librosa
import soundfile as sf
import shutil
import string
from random import shuffle
import re
from shutil import copy2
from utils import normalize, absoluteFilePaths

def convert_to_16k(path):
    in_path = path
    out_path = os.path.dirname(path)
    file_name = ntpath.basename(in_path)
    y, s = librosa.load(in_path, sr=16000)
    y_16k = librosa.resample(y, s, 16000)
    path_to_write = os.path.join(out_path,file_name)
    sf.write(path_to_write, y_16k, 16000, format='WAV', subtype='PCM_16')

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_file", default=None, type=str,
                        required=True, help="Path to train file, eg. train.lst")
    
    parser.add_argument("--test_file", default=None, type=str,
                        required=True, help="Path to test file, eg. test.lst")
    
    parser.add_argument("--audio_path", default=None, type=str,
                        required=True, help="Path to wav files, eg. audio")
    
    parser.add_argument("--wav2vec_file", default=None, required=True,
                        type=str,help="Path to wav2vec model, check in resources/wav2vec.pt")
    
    parser.add_argument("--wav2letter", default=None, type=str,
                        required=True, help="Path to wav2letter library")
    
    parser.add_argument("--am_file", default=None, type=str,
                        help="Path to base aucostic model. If this is not given --> Training from scratch")
    
    parser.add_argument("--arch_file", default=None, type=str,
                        required=True, help="Path to archtecture file, check in resources/network.arch")
    
    parser.add_argument("--token_file", default=None, type=str,
                        required=True, help="Path to token file, check in resources/tokens.txt")
    
    parser.add_argument("--lexicon_file", default=None, type=str,
                        required=True, help="Path to lexicon file, check in resources/lexicon.txt")
    
    parser.add_argument("--output_path", default=None, type=str,
                        required=True, help="Output path for storing feature and model")
    
    parser.add_argument("--mode", default=None, type=str, required=True, 
                        help="Either 'finetune' or 'scratch'" 
                             "If scratch  --> train AM from scratch"
                             "If finetune --> continue training using AM provided by arch_file")
    
    parser.add_argument("--iter", default=50, type=int,
                        help="Number of interations. Set this number higher if training from scratch")
    
    parser.add_argument("--lr", default=0.5, type=float,
                        help="Learning rate, set 1.0 for training from scratch and 0.5 for fine-tunning")
    
    parser.add_argument("--lrcrit", default=0.004, type=float,
                        help="Learning rate crit, set 0.006 for training from scratch and 0.001 for fine-tunning ")
    
    parser.add_argument("--momentum", default=0.5, type=float,help="SGD momentum")
    
    parser.add_argument("--maxgradnorm", default=0.05, type=float,help="Max gradnorm")
    
    parser.add_argument("--nthread", default=1, type=int,help="Number of jobs")
    
    args = parser.parse_args()
    
    if args.mode not in ['scratch','finetune']:
        raise ValueError('Training mode must be either scratch or finetune')
    
    if args.mode == 'finetune' and args.am_file == None:
        raise ValueError('For finetune training, am_file must be given')
    
    if args.mode == 'scratch':
        args.lr = 1.0
        args.lrcrit = 0.01
        args.iter = 100
        args.momentum = 0.8
        args.maxgradnorm = 0.1
    
    print("Converting to 16k !!!")
    audio_names = absoluteFilePaths(args.audio_path)
    pool = Pool(4)
    pool.map(convert_to_16k,audio_names)
    pool.terminate()

    w2vec = Prediction(args.wav2vec_file)

    #Extract wav2vec feature
    featureWritter = EmbeddingDatasetWriter(input_root = args.audio_path,
                                            output_root = os.path.join(args.output_path,'feature'),
                                            loaded_model = w2vec, 
                                            extension="wav",use_feat=False)
    featureWritter.write_features()

    #Write train file
    feature_path = os.path.join(args.output_path,'feature')

    with open(args.train_file) as f:
        data = f.read().split('\n')
        data = [t for t in data if len(t) > 1]
        data = [d.split('\t') for d in data]

    for i in range(0,len(data)):
        path = os.path.join(args.audio_path, data[i][0])
        text = normalize(data[i][1])
        audio = AudioSegment.from_wav(path)
        path = os.path.join(feature_path,ntpath.basename(path))
        path = os.path.abspath(path)
        path = path.replace('.wav','.h5context')
        leng = str(len(audio) / 1000.0)
        idx = 'train' + str(i)
        data[i] = '\t'.join([idx, path, leng, text])

    train_feature_file = abspath(os.path.join(args.output_path, 'train.lst'))
    with open(train_feature_file, 'w') as f:
        f.write('\n'.join(data))

    #Write test file
    with open(args.test_file) as f:
        data = f.read().split('\n')
        data = [t for t in data if len(t) > 1]
        data = [d.split('\t') for d in data]

    for i in range(0,len(data)):
        path = os.path.join(args.audio_path, data[i][0])
        text = normalize(data[i][1])
        audio = AudioSegment.from_wav(path)
        path = os.path.join(feature_path,ntpath.basename(path))
        path = os.path.abspath(path)
        path = path.replace('.wav','.h5context')
        leng = str(len(audio) / 1000.0)
        idx = 'test' + str(i)
        data[i] = '\t'.join([idx, path, leng, text])

    test_feature_file = abspath(os.path.join(args.output_path, 'test.lst'))
    with open(test_feature_file, 'w') as f:
        f.write('\n'.join(data))
    
    tokens = ntpath.basename(args.token_file)
    tokendirs = os.path.dirname(abspath(args.token_file))
    arch = ntpath.basename(args.arch_file)
    archdirs = os.path.dirname(abspath(args.arch_file))
    
    cmd = ['--runname=model']
    cmd.append('--rundir=' + abspath(args.output_path))
    cmd.append('--tokensdir=' + tokendirs)
    cmd.append('--tokens=' + tokens)
    cmd.append('--lexicon=' + args.lexicon_file)
    cmd.append('--archdir=' + archdirs)
    cmd.append('--arch=' + arch)
    cmd.append('--train=' + train_feature_file)
    cmd.append('--valid=' + test_feature_file)
    cmd.append('--lr=' + str(args.lr))
    cmd.append('--lrcrit=' + str(args.lrcrit))
    cmd.append('--iter=' + str(args.iter))
    cmd.append('--momentum=' + str(args.momentum))
    cmd.append('--maxgradnorm=' + str(args.maxgradnorm))
    cmd.append('--input=hdf5')
    cmd.append('--criterion=asg')    
    cmd.append('--linseg=1')
    cmd.append('--replabel=2')
    cmd.append('--onorm=target')
    cmd.append('--wnorm=true')
    cmd.append('--surround=|')
    cmd.append('--sqnorm=true')
    cmd.append('--mfsc=false')
    cmd.append('--wav2vec=true')
    cmd.append('--nthread=1')
    cmd.append('--batchsize=4')
    cmd.append('--transdiag=5')
    cmd.append('--melfloor=1.0')
    cmd.append('--minloglevel=0')
    cmd.append('--logtostderr=1')
    cmd.append('--enable_distributed=false')
    
    cfg_path = os.path.join(args.output_path, 'fork_vec.cfg')
    with open(cfg_path,'w') as f:
        f.write('\n'.join(cmd))
    
    cmd = ['sudo']
    cmd.append(os.path.join(args.wav2letter,'build/Train'))
    
    if args.mode == 'finetune':
        cmd.append('fork ' + abspath(args.am_file))
    else:
        cmd.append('train ')
    
    cmd.append('--flagsfile=' + cfg_path)
    cmd = ' '.join(cmd)
    print(cmd)
    time.sleep(5)
    os.system(cmd)
    
    path_to_model = os.path.join(args.output_path, 'model/001_model_' + test_feature_file.replace('/','#') + '.bin')
    path_to_write = os.path.join(args.output_path, 'am.bin')
    copy2(path_to_model,path_to_write)
    
    os.system('sudo rm -rf ' + os.path.join(args.output_path,'model'))
    shutil.rmtree(os.path.join(args.output_path,'feature'), ignore_errors=True)
    os.remove(test_feature_file)
    os.remove(train_feature_file)
    os.remove(cfg_path)

main()
