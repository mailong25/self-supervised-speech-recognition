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

#### 1. Train a self-supervised model on unlabeled data

#### 2. Train a language model

#### 3. Finetune the self-supervised model on the labeled data

#### 4. Make prediction on single audio

## Older version on Vietnamese speech recognition: 
https://github.com/mailong25/self-supervised-speech-recognition/tree/vietnamese
