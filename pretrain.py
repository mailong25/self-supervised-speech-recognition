import argparse
import os
from os.path import join as join_path
import torch
import multiprocessing
import sys

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fairseq_path", default='/mnt/disks/', type=str,
                        required=False, help="Path to installed fairseq library")
    
    parser.add_argument("--audio_path", default='/mnt/disks/', type=str,
                        required=False, help="Path to unlabeled audio")
    
    parser.add_argument("--init_model", default='mnt/disks/a.txt', required=False,
                        type=str,help="Path to English pretrain wav2vec model")
    
    args = parser.parse_args()
    
    #Prepare manifest file
    MANIFEST_PATH = join_path(args.fairseq_path, 'examples/wav2vec/wav2vec_manifest.py')
    cmd = 'python3 ' + MANIFEST_PATH + ' ' + args.audio_path + ' --dest temp --ext wav --valid-percent 0.05'
    os.system(cmd)
    
    #Pretrain the model
    NUM_GPU = torch.cuda.device_count()
    NUM_CPU = multiprocessing.cpu_count()
    
    if NUM_GPU == 0:
        print("pytorch cannot find any GPUs !")
        sys.exit(0)
    
    cmd = "fairseq-hydra-train task.data=temp distributed_training.distributed_world_size=" + str(NUM_GPU) + " +optimization.update_freq='[" + str(int(64/NUM_GPU)) + "]' checkpoint.finetune_from_model=" + args.init_model + " dataset.num_workers=" + str(NUM_CPU) + " --config-dir config/pretraining --config-name wav2vec2_base_librispeech"
    
    os.system(cmd)
    
main()