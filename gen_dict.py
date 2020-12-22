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

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--transcript_file", default=None, type=str,
                        required=True, help="Path to transcript file")
    
    parser.add_argument("--save_dir", default=None, required=True,
                        type=str,help="Directory to save dictionary file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    dictionary = os.path.join(args.save_dir,'dict.ltr.txt')
    
    with open(args.transcript_file) as f:
        data = f.read().splitlines()
    
    words = [d.split('\t')[1].upper() for d in data]
    letters = [d.replace(' ','|') for d in words]
    letters = [' '.join(list(d)) + ' |' for d in letters]

    chars = [l.split() for l in letters]
    chars = [j for i in chars for j in i]
    char_stats = list(Counter(chars).items())
    char_stats = sorted(char_stats, key=lambda x : x[1], reverse = True)
    char_stats = [c[0] + ' ' + str(c[1]) for c in char_stats]
    
    with open(dictionary,'w') as f:
        f.write('\n'.join(char_stats))
    
main()