## State-of-the-art speech recognition with Wav2Vec and Wav2letter
This repository contains ready-to-use software for Vietnamese automatic speech recognition. After downloading pre-trained models and installing dependencies, you can quickly make predictions by using:

```
from stt import Transcriber

transcriber = Transcriber(w2letter = '/path/to/wav2letter', w2vec = 'resources/wav2vec.pt', 
                          am = 'resources/am.bin', tokens = 'resources/tokens.txt', 
                          lexicon = 'resources/lexicon.txt', lm = 'resources/lm.bin',
                          temp_path = './temp',nthread_decoder = 4)

transcriber.transcribe(['data/audio/VIVOSSPK01_R001.wav','data/audio/VIVOSSPK01_R002.wav'])
```
Where '/path/to/wav2letter' refer to the actual path to wav2letter library after you've done installing [dependencies](https://github.com/mailong25/vietnamese-speech-recognition/blob/master/dependencies.md). The remaining parameters can be found in resources directory.


## Key Techniques:
 - [Speech2vec](https://arxiv.org/abs/1904.05862) using self-supervised learning to extract representations of raw audio. The model is trained on large amounts of unlabeled audio data (500hours), and then used to improve acoustic model training. As a result, it significantly outperforms traditional MFCC features in a low-resource setting.
 - [wav2letter](https://arxiv.org/pdf/1609.03193.pdf): for training Acoustic Modeling
 - [kenlm](https://github.com/kpu/kenlm): for training Language Modeling.

Pipeline:

Audio -> Feature extraction (Wav2Vec) -> Acoustic Modeling (wav2letter) -> Language Model (LM) -> Decoding --> Texts


## Install dependencies
Warning: This might take a lot of time and effort, so keep calm and stick to the end.

Please follow this [instruction](https://github.com/mailong25/vietnamese-speech-recognition/blob/master/dependencies.md)


## Pre-trained models
Download pre-trained models, including Wav2vec, AM, and LM at this [link](https://drive.google.com/file/d/1q7ReoRT9yeDxVm8Xj521n-c-bIhgcBwU/view?usp=sharing). After that, put all files into resources directory

Wav2vec model are trained on 500 hours of unlabeled data, while AM are trained on ~70 hours of labeled audio.

Current WER is about 15%

## Continue training
If you have domain-specific labeled data, You can finetune our model on your dataset (domain adaptation):
```
python3 train.py --train_file data/train.lst --test_file data/test.lst \
--audio_path data/audio --wav2vec_file resources/wav2vec.pt \
--wav2letter /path/to/wav2letter --am_file resources/am.bin \
--arch_file resources/network.arch --token_file resources/tokens.txt \
--lexicon_file resources/lexicon.txt --output_path out --mode=finetune
```

## Further improvements
Check out this great work from Facebook research:

Effectiveness of self-supervised pre-training for speech recognition. Alexei Baevski, Michael Auli, Abdelrahman Mohamed, [arxiv](https://arxiv.org/abs/1911.03912)

> We propose a BERT-style model learning directly from the continuous audio data and compare pre-training on raw audio to spectral features. Fine-tuning a BERT model on 10 hour of labeled Librispeech data with a vq-wav2vec vocabulary is almost as good as the best known reported system trained on 100 hours of labeled data on testclean, while achieving a 25% WER reduction on test-other. When using only 10 minutes of labeled data, WER is 25.2 on test-other and 16.3 on test-clean. This demonstrates that self-supervision can enable speech recognition systems trained on a near-zero amount of transcribed data.
