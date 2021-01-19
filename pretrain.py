import argparse
import os
from os.path import join as join_path
import torch
import multiprocessing
import sys

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fairseq_path", default=None, type=str,
                        required=True, help="Path to installed fairseq library")
    
    parser.add_argument("--audio_path", default=None, type=str,
                        required=True, help="Path to unlabeled audio")
    
    parser.add_argument("--init_model", default=None, required=False,
                        type=str,help="Path to English pretrain wav2vec model")
    
    parser.add_argument("--batch_size", default=1200000, required=False,
                        type=int,help="Batch size, try to decrease this number if any CUDA memory problems occur")
    
    args = parser.parse_args()
    
    #Prepare manifest file
    MANIFEST_PATH = join_path(args.fairseq_path, 'examples/wav2vec/wav2vec_manifest.py')
    
    temp_dir = os.path.abspath('./temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    cmd = 'python3 ' + MANIFEST_PATH + ' ' + args.audio_path + ' --dest ' + temp_dir + ' --ext wav --valid-percent 0.05'
    os.system(cmd)
    
    #Pretrain the model
    NUM_GPU = torch.cuda.device_count()
    NUM_CPU = multiprocessing.cpu_count()
    
    if NUM_GPU == 0:
        print("pytorch cannot find any GPUs !")
        sys.exit(0)
    
    cmd = ["fairseq-hydra-train"]
    cmd.append("task.data=" + str(temp_dir))
    cmd.append("distributed_training.distributed_world_size=" + str(NUM_GPU))
    cmd.append("+optimization.update_freq='[" + str(int(64/NUM_GPU)) + "]'")
    
    if args.init_model != None:
        cmd.append("checkpoint.finetune_from_model=" + os.path.abspath(args.init_model))
    
    cmd.append("dataset.num_workers=" + str(NUM_CPU))
    cmd.append("dataset.max_tokens=" + str(args.batch_size))
    cmd.append("--config-dir config/pretraining")
    cmd.append("--config-name wav2vec2_base_librispeech")
    cmd = ' '.join(cmd)
    print(cmd)
    
    os.system(cmd)
    
main()
