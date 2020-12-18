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

## Steps to build an accurate speech recognition model for your language

#### 1. Train a self-supervised model on unlabeled data

#### 2. Train a language model

#### 3. Finetune the self-supervised model on the labeled data

#### 4. Make prediction on single audio

## Install dependencies

#### 0. Create a folder to store external libs
```
mkdir libs
cd libs
```

#### 1. Install python package
```
pip install soundfile torchaudio sentencepiece
If cuda version < 11 (eg. cuda 10.1):
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
If cuda >= 11:
pip install torch==1.6.0
```

#### 2. Install fairseq
```
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout c8a0659be5cdc15caa102d5bbf72b872567c4859
pip install --editable ./
cd ..
```

#### 3. Install wav2letter decoder bindings
```
sudo apt-get update && apt-get upgrade -y && apt-get install -y && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
sudo apt-get install libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
```

#### 4. Install kenlm
```
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake ..
make -j 4
cd ../..
```

# 5. Install wav2letter
```
git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
cd wav2letter/bindings/python
export KENLM_ROOT_DIR=path/to/libs/kenlm/ && pip install -e .
cd ../../..
```

## Older version on Vietnamese speech recognition: 
https://github.com/mailong25/self-supervised-speech-recognition/tree/vietnamese
