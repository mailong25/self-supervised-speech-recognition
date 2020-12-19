from os.path import abspath
import os
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--kenlm_path", default=None, type=str,
                        required=True, help="Path to kenlm library")
    
    parser.add_argument("--text_file", default=None, type=str,required=True,
                        help="Path to general text (optional) , check in resources/general_text.txt")
    
    parser.add_argument("--output_path", default=None, type=str,
                        required=True, help="Output path for storing model")
    
    args = parser.parse_args()
    
    with open(args.text_file) as f:
        train = f.read().upper().splitlines()
    
    vocabs = set([])
    for line in train:
        vocabs = vocabs | set(line.split())    
    vocabs = list(vocabs)
    
    vocab_path = os.path.join(args.output_path,'vocabs.txt')
    lexicon_path = os.path.join(args.output_path,'lexicon.txt')
    train_text_path = os.path.join(args.output_path,'world_lm_data.train')
    train_text_path_train = train_text_path.replace('world_lm_data.train','kenlm.train')
    model_arpa = train_text_path.replace('world_lm_data.train','kenlm.arpa')
    model_bin  = train_text_path.replace('world_lm_data.train','lm.bin')
    kenlm_path_train = os.path.join(abspath(args.kenlm_path) , 'build/bin/lmplz')
    kenlm_path_convert = os.path.join(abspath(args.kenlm_path) , 'build/bin/build_binary')
    kenlm_path_query = os.path.join(abspath(args.kenlm_path) , 'build/bin/query')
    
    with open(train_text_path,'w') as f:
        f.write('\n'.join(train))
        
    with open(vocab_path,'w') as f:
        f.write(' '.join(vocabs))
    
    for i in range(0,len(vocabs)):
        vocabs[i] = vocabs[i] + '\t' + ' '.join(list(vocabs[i])) + ' |'
        
    with open(lexicon_path,'w') as f:
        f.write('\n'.join(vocabs))
    
    cmd = kenlm_path_train + " -T /tmp -S 4G --discount_fallback -o 4 --limit_vocab_file " + vocab_path + " trie < " + train_text_path +  ' > ' + model_arpa
    os.system(cmd)
    cmd = kenlm_path_convert +' trie ' + model_arpa + ' ' + model_bin
    os.system(cmd)
    cmd = kenlm_path_query + ' ' + model_bin + " < " + train_text_path + ' > ' + train_text_path_train
    os.system(cmd)
    os.remove(train_text_path)
    os.remove(train_text_path_train)
    os.remove(model_arpa)
    os.remove(vocab_path)

main()