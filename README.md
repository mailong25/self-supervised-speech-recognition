## Self-supervised speech recognition with limited amount of labeled data


This is a wrapper version of [wav2vec 2.0 framework](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec), which attempts to build an accurate speech recognition models with small amount of transcribed data (eg. 1 hour)


Transfer learning is still the main technique:
 - Transfer from self-supervised models (pretrain on unlabeled data)
 - Transfer from multilingual models (pretrain on multilingual data)

## Required resources

#### 1. Labeled data, which is pairs of (audio, transcript)
The more you have, the better the model is. Prepare at least 1 hour if you have a large amount of  unlabeled data. Otherwise, at least 50 hours is recommended.

#### 2. Text data for building language models. 
This should includes both well-written text and conversational text, which can easily collected from news/forums websties. At least 1 GB of text is recommended.

#### 3. Unlabeled data (audios without transcriptions) of your own language. 
This is optional but very crucial. A good amount of unlabeled audios (eg. 500 hours) will significantly reduce the amount of labeled data needed, and also boost up the model performance. Youtube/Podcast is a great place to collect the data for your own language

## Install instruction
Please follow this [instruction](https://github.com/mailong25/self-supervised-speech-recognition/blob/master/Dependencies.md)

## Steps to build an accurate speech recognition model for your language

### 1. Train a self-supervised model on unlabeled data (Pretrain)

#### 1.1 Prepare unlabeled audios
Collect unlabel audios and put them all together in a single directory. Audio format requirements:\
Format: wav, PCM 16 bit, single channel\
Sampling_rate: 16000\
Length: 5 to 30 seconds\
Content: silence should be removed from the audio. Also, each audio should contain only one person speaking.\
Please look at examples/unlabel_audio directory for reference.

#### 1.2 Download an initial model
Instead of training from scratch, we download and use english wav2vec model for weight initialization. This pratice can be apply to all languages.
```
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
```

#### 1.3 Run Pre-training
```
python3 pretrain.py --fairseq_path path/to/libs/fairseq --audio_path path/to/audio_directory --init_model path/to/wav2vec_small.pt
```
Where:
 - fairseq_path: path to installed fairseq library, after install [instruction](https://github.com/mailong25/self-supervised-speech-recognition/blob/master/Dependencies.md)
 - audio_path: path to unlabel audio directory
 - init_model: downloaded model from step 1.2

Logs and checkpoints will be stored at outputs directory\
Log_file path: outputs/date_time/exp_id/hydra_train.log.  You should check the loss value to decide when to stop the training process.\
Best_checkpoint path: outputs/date_time/exp_id/checkpoints/checkpoint_best.pt\
In my casse, it took ~ 4 days for the model to converge, train on 100 hours of data using 2 NVIDIA Tesla V100.

### 2. Finetune the self-supervised model on the labeled data

#### 2.1 Prepare labeled data
-- Transcript file ---\
One trainng sample per line with format "audio_absolute_path \tab transcript"\
Example of a transcript file:
```
/path/to/1.wav AND IT WAS A MATTER OF COURSE THAT IN THE MIDDLE AGES WHEN THE CRAFTSMEN
/path/to/2.wav AND WAS IN FACT THE KIND OF LETTER USED IN THE MANY SPLENDID MISSALS PSALTERS PRODUCED BY PRINTING IN THE FIFTEENTH CENTURY
/path/to/3.wav JOHN OF SPIRES AND HIS BROTHER VINDELIN FOLLOWED BY NICHOLAS JENSON BEGAN TO PRINT IN THAT CITY
/path/to/4.wav BEING THIN TOUGH AND OPAQUE
```
Some notes on transcript file:
- One sample per line
- Upper case
- All numbers should be transformed into verbal form.
- All special characters (eg. punctuation) should be removed. The final text should contain words only
- Words in a sentence must be separated by whitespace character


-- Labeled audio file ---\
Format: wav, PCM 16 bit, single channel, Sampling_rate: 16000.\
Silence should be removed from the audio.\
Also, each audio should contain only one person speaking.\

#### 2.2 Generate dictionary file
```
python3 gen_dict.py --transcript_file path/to/transcript.txt --save_dir path/to/save_dir
```
The dictionary file will be stored at save_dir/dict.ltr.txt. Use the file for fine-tuning and inference.

#### 2.3 Run Fine-tuning on the pretrain model
```
python3 finetune.py --transcript_file path/to/transcript.txt --pretrain_model path/to/pretrain_checkpoint_best.pt --dict_file path/to/dict.ltr.txt
```
Where:
 - transcript_file: path to transcript file from step 2.1
 - pretrain_model: path to best model checkpoint from step 1.3
 - dict_file: dictionary file generated from step 2.2

Logs and checkpoints will be stored at outputs directory\
Log_file path: outputs/date_time/exp_id/hydra_train.log. You should check the loss value to decide when to stop the training process.\
Best_checkpoint path: outputs/date_time/exp_id/checkpoints/checkpoint_best.pt\
In my casse, it took ~ 12 hours for the model to converge, train on 100 hours of data using 2 NVIDIA Tesla V100.

### 3. Train a language model
#### 3.1 Prepare text corpus
Collect all texts and put them all together in a single file. \
Text file format:
- One sentence per line
- Upper case
- All numbers should be transformed into verbal form.
- All special characters (eg. punctuation) should be removed. The final text should contain words only
- Words in a sentence must be separated by whitespace character

Example of a text corpus file for English case:
```
AND IT WAS A MATTER OF COURSE THAT IN THE MIDDLE AGES WHEN THE CRAFTSMEN
AND WAS IN FACT THE KIND OF LETTER USED IN THE MANY SPLENDID MISSALS PSALTERS PRODUCED BY PRINTING IN THE FIFTEENTH CENTURY
JOHN OF SPIRES AND HIS BROTHER VINDELIN FOLLOWED BY NICHOLAS JENSON BEGAN TO PRINT IN THAT CITY
BEING THIN TOUGH AND OPAQUE
...
```
Example of a text corpus file for Chinese case:
```
每 个 人 都 有 他 的 作 战 策 略 直 到 脸 上 中 了 一 拳
这 是 我 年 轻 时 候 住 的 房 子 。
这 首 歌 使 我 想 起 了 我 年 轻 的 时 候 。
...
```

#### 3.2 Train the language model
```
python3 train_lm.py --kenlm_path path/to/libs/kenlm --transcript_file path/to/transcript.txt --additional_file path/to/text_corpus.txt --ngram 3 --output_path ./lm
```
Where:
 - kenlm_path: path to installed kenlm library, after install [instruction](https://github.com/mailong25/self-supervised-speech-recognition/blob/master/Dependencies.md)
 - transcript_file: path to transcript file from step 2.1
 - additional_file: path to text corpus file from step 3.1

The LM model and the lexicon file will be stored at output_path

### 4. Make prediction on multiple audios programmatically

```
from stt import Transcriber
transcriber = Transcriber(pretrain_model = 'path/to/pretrain.pt', finetune_model = 'path/to/finetune.pt', 
                          dictionary = 'path/to/dict.ltr.txt',
                          lm_type = 'kenlm',
                          lm_lexicon = 'path/to/lm/lexicon.txt', lm_model = 'path/to/lm/lm.bin',
                          lm_weight = 1.5, word_score = -1, beam_size = 50)
hypos = transcriber.transcribe(['path/to/wavs/0_1.wav','path/to/wavs/0_2.wav'])
print(hypos)
```

Where:
 - pretrain_model: path to best pretrain checkpoint from step 1.3 
 - finetune_model: path to best fine-tuned checkpoint from step 2.3
 - dictionary: dictionary file generated from step 2.2
 - lm_lexicon and lm_model: generated from step 3.2

Note: If you running inference in a juyter notebook. Please add these lines above the inference script:
```
import sys
sys.argv = ['']
```


## Pre-trained models (Pretrain + Fine-tune + LM)
- [Vietnamese](https://drive.google.com/file/d/1kZFdvMQt-R7fVebTbfWMk8Op7I9d24so/view?usp=sharing)


## Older version on Vietnamese speech recognition: 
https://github.com/mailong25/self-supervised-speech-recognition/tree/vietnamese

## Reference:
Paper: wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations: https://arxiv.org/abs/2006.11477 \
Source code: https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md

## License
MIT
