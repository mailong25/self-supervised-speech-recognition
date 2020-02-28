from __future__ import absolute_import, division, print_function
import soundfile as sf
import subprocess
import os, sys
import numpy as np
import ntpath
import sh
import time
from wav2vec import EmbeddingDatasetWriter
from wav2vec import Prediction
from utils import absoluteFilePaths, convert_to_16k
from utils import silenceRemovalWrapper, chunk_audio, read_result
from pydub import AudioSegment
from multiprocessing import Pool

def preprocessing(args):
    file_path  = args[0]
    file_index = args[1]
    output_path = args[2]
    new_file_path = os.path.join(output_path, str(file_index) + '.wav')
    audio = AudioSegment.from_wav(file_path)
    convert_to_16k(file_path, new_file_path)
    chunk_audio(new_file_path, output_path, max_len = 12)

class Transcriber:
    def __init__(self, w2letter, w2vec, am, tokens, lexicon, lm,
                 nthread_decoder = 1, lmweight = 1.51, wordscore = 2.57, beamsize = 200,
                 temp_path = './temp'):
        '''
        w2letter : path to complied wav2letter library (eg. /home/wav2letter)
        w2vec    : path to wav2vec model
        am       : path to aucostic model
        tokens   : path to graphmemes file
        lexicon  : path to dictionary file
        lm       : path to language model
        nthread_decoder: number of jobs for speeding up
        lmweight  : how much language model affect the result, the higher the more important
        wordscore : weight score for group of letter forming a word
        beamsize  : number of path for decoding, the higher the better but slower
        temp_path : directory for storing temporary files during processing
        '''
        
        self.w2letter = os.path.abspath(w2letter)
        self.am = os.path.abspath(am)
        self.tokens = ntpath.basename(tokens)
        self.tokensdir = os.path.dirname(os.path.abspath(tokens))
        self.lexicon = os.path.abspath(lexicon)
        self.lm = os.path.abspath(lm)
        self.nthread_decoder = nthread_decoder
        self.lmweight = lmweight
        self.wordscore = wordscore
        self.beamsize = beamsize
        self.pool = Pool(nthread_decoder)
        self.w2vec = Prediction(w2vec)
        self.output_path = os.path.abspath(temp_path)
        print(self.__dict__)
        
    def decode(self, input_file, output_path):
        cmd = ['sudo']
        cmd.append(os.path.join(self.w2letter,'build/Decoder'))
        cmd.append('--am=' + self.am)
        cmd.append('--tokensdir=' + self.tokensdir)
        cmd.append('--tokens=' + self.tokens)
        cmd.append('--lexicon=' + self.lexicon)
        cmd.append('--lm=' + self.lm)
        cmd.append('--test=' + input_file)
        cmd.append('--sclite=' + str(output_path))
        cmd.append('--lmweight=' + str(self.lmweight))
        cmd.append('--wordscore=' + str(self.wordscore))
        cmd.append('--beamsize=' + str(self.beamsize))
        cmd.append('--beamthreshold=50')
        cmd.append('--silweight=-0.595')
        cmd.append('--nthread_decoder=' + str(self.nthread_decoder))
        cmd.append('--smearing=max')
        cmd.append('--lm_memory=3000')
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        process.wait()
        process.kill()
        return ' '.join(cmd)
        
    def transcribe(self,wav_files):
        
        start = time.time()
        self.pool.map(preprocessing, [(wav_files[i], i, self.output_path) for i in range(0,len(wav_files))])
        print("Preprocessing: ", time.time() - start)
        start = time.time()
        
        #Extract wav2vec feature
        featureWritter = EmbeddingDatasetWriter(input_root = self.output_path,
                                                output_root = self.output_path,
                                                loaded_model = self.w2vec, 
                                                extension="wav",use_feat=False)
        featureWritter.write_features()

        print("Feature extraction: ", time.time() - start)
        start = time.time()
        
        #Prepare dataset for speech to text
        paths = absoluteFilePaths(self.output_path)
        paths = [p for p in paths if '.h5context' in p]
        lines = []
        for p in paths:
            file_name = ntpath.basename(p).replace('.h5context','')
            lines.append('\t'.join([file_name, p, '5', 'anh em']))

        with open(os.path.join(self.output_path, 'test.lst'),'w') as f:
            f.write('\n'.join(lines))

        #Decoding on created dataset
        decode_res = self.decode(os.path.join(self.output_path, 'test.lst'),self.output_path)

        print("Decoding: ", time.time() - start)
        
        trans_file = None
        for path in absoluteFilePaths(self.output_path):
            if 'test.lst.hyp' in path:
                trans_file = path

        if trans_file == None:
            print("An error occurs during decoding. Please run the following command line in a seperate terminal :")
            print(decode_res)
        
        transcripts = read_result(trans_file)
        transcripts = list(transcripts.items())
        transcripts = sorted(transcripts, key = lambda x : x[0])
        transcripts = [t[1] for t in transcripts]
        
        sh.rm(sh.glob(self.output_path + '/*'))
        
        return transcripts