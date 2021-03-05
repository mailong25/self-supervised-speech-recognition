## Install dependencies
```
git clone https://github.com/mailong25/self-supervised-speech-recognition.git
cd self-supervised-speech-recognition
```

#### 0. Create a folder to store external libs
```
mkdir libs
cd libs
```

#### 1. Install python package
```
pip install soundfile torchaudio sentencepiece editdistance sklearn
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

#### 3. Install dependencies for wav2letter
```
sudo apt-get update && sudo apt-get -y install apt-utils libpq-dev libsndfile-dev
sudo apt install libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
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

#### 5. Install wav2letter decoder bindings
```
git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
cd wav2letter/bindings/python
export KENLM_ROOT_DIR=path/to/libs/kenlm/ && pip install -e .
cd ../../..
```
