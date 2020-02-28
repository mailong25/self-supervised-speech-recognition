## Vietnamese speech recognition with Wav2Vec and Wav2letter
This repository contains ready-to-use software for automatic speech recognition. After downloading pre-trained models and installing dependencies, you can quickly make predictions by using:

```
from stt import Transcriber

transcriber = Transcriber(w2letter = '/path/to/wav2letter', w2vec = 'resources/wav2vec.pt', 
                          am = 'resources/am.bin', tokens = 'resources/tokens.txt', 
                          lexicon = 'resources/lexicon.txt', lm = 'resources/lm.bin',
                          temp_path = './temp',nthread_decoder = 4)

transcriber.transcribe(['data/audio/VIVOSSPK01_R001.wav','data/audio/VIVOSSPK01_R002.wav'])
```
Where '/path/to/wav2letter' refer to the actual path to wav2letter library after you've done installing [dependencies](https://github.com/mailong25/vietnamese-speech-recognition/blob/master/dependencies.md). The remaining parameters can be found in resources directory.


## Technical overview
Techniques and softwares were used to build the model:
 - [Speech2vec](https://arxiv.org/abs/1904.05862) using self-supervised learning to extract representations of raw audio. The model is trained on large amounts of unlabeled audio data (500hours), and then used to improve acoustic model training. As a result, it significantly outperforms traditional MFCC features in a low-resource setting.
 - [wav2letter](https://arxiv.org/pdf/1609.03193.pdf): for training Acoustic Modeling
 - [kenlm](https://github.com/kpu/kenlm): for training Language Modeling.

Pipeline:

Audio -> Feature extraction (Wav2Vec) -> Acoustic Modeling (wav2letter) -> Language Model (LM) -> Decoding --> Texts


## Install dependencies
Warning: This might take a lot of time and effort, so be ready for it and stick to the end.

Please follow this [instruction](https://github.com/mailong25/vietnamese-speech-recognition/blob/master/dependencies.md)


## Pre-trained models
Download pre-trained models, including Wav2vec, AM, and LM at this [link](https://drive.google.com/file/d/1q7ReoRT9yeDxVm8Xj521n-c-bIhgcBwU/view?usp=sharing). After that, put all files into resources directory


