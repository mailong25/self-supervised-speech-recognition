import argparse
import os
from os.path import join as join_path
import torch
import multiprocessing
import sys
from collections import Counter
from tqdm import tqdm
from sklearn.utils import shuffle
import ntpath
import soundfile
from shutil import copy2

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--transcript_file", default=None, type=str,
                        required=True, help="Path to transcript file")
    
    parser.add_argument("--pretrain_model", default=None, required=True,
                        type=str,help="Path to pretrain wav2vec model")
    
    parser.add_argument("--dict_file", default=None, required=True,
                        type=str,help="Path to dictionary file")
    
    args = parser.parse_args()
    
    args.pretrain_model = os.path.abspath(args.pretrain_model)
    args.save_dir = os.path.abspath('./manifest')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    copy2(args.dict_file,args.save_dir)
    
    #Pretrain the model
    NUM_GPU = torch.cuda.device_count()
    NUM_CPU = multiprocessing.cpu_count()
    
    if NUM_GPU == 0:
        print("pytorch cannot find any GPUs !")
        sys.exit(0)
    
    # Create manifest files
    train_words = os.path.join(args.save_dir,'train.wrd')
    valid_words = os.path.join(args.save_dir,'valid.wrd')
    train_letters = os.path.join(args.save_dir,'train.ltr')
    valid_letters = os.path.join(args.save_dir,'valid.ltr')
    train_map = os.path.join(args.save_dir,'train.tsv')
    valid_map = os.path.join(args.save_dir,'valid.tsv')
    
    with open(args.transcript_file) as f:
        data = f.read().splitlines()
    
    words = [d.split('\t')[1].upper() for d in data]
    letters = [d.replace(' ','|') for d in words]
    letters = [' '.join(list(d)) + ' |' for d in letters]
    
    paths = [d.split('\t')[0] for d in data]
    total_duration = 0
    
    for i in tqdm(range(0,len(paths))):
        audio_info = soundfile.info(paths[i])
        frames = audio_info.frames
        total_duration += audio_info.duration
        paths[i] = paths[i] + '\t' + str(frames)
    
    SPLIT_NUM = int(len(words)*0.95)
    
    words,letters,paths = shuffle(words,letters,paths)
    
    train_w, valid_w = words[:SPLIT_NUM], words[SPLIT_NUM:]
    train_l, valid_l = letters[:SPLIT_NUM], letters[SPLIT_NUM:]
    train_p, valid_p = paths[:SPLIT_NUM], paths[SPLIT_NUM:]
    
    with open(train_words,'w') as f:
        f.write('\n'.join(train_w))
        
    with open(valid_words,'w') as f:
        f.write('\n'.join(valid_w))
    
    with open(train_letters,'w') as f:
        f.write('\n'.join(train_l))
        
    with open(valid_letters,'w') as f:
        f.write('\n'.join(valid_l))
        
    with open(train_map,'w') as f:
        f.write('\n')
        f.write('\n'.join(train_p))
    
    with open(valid_map,'w') as f:
        f.write('\n')
        f.write('\n'.join(valid_p))
    
    total_duration = total_duration / 3600.0
    
    if total_duration <= 5:
        config_name = "base_1h"
    elif total_duration <= 50:
        config_name = "base_10h"
    elif total_duration <= 500:
        config_name = "base_100h"
    else:
        config_name = "base_960h"
    
    cmd = "fairseq-hydra-train task.data=" + str(args.save_dir) + " distributed_training.distributed_world_size=" + str(NUM_GPU) + " +optimization.update_freq='[" + str(int(24/NUM_GPU)) + "]' model.w2v_path=" + args.pretrain_model + " dataset.num_workers=" + str(NUM_CPU) + " --config-dir config/finetuning --config-name " + config_name
    
    print(cmd)
    
    os.system(cmd)
    
main()