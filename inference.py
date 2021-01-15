from stt import Transcriber
import sys
import configparser
import glob
import os

def main():
	config = configparser.ConfigParser()
	config.read("config.txt")
	transcriber = Transcriber(pretrain_model = config["TRANSCRIBER"]["pretrain_model"],
							finetune_model = config["TRANSCRIBER"]["finetune_model"],
							dictionary = config["TRANSCRIBER"]["dictionary"],
							lm_lexicon = config["TRANSCRIBER"]["lm_lexicon"],
							lm_model = config["TRANSCRIBER"]["lm_model"],
							lm_weight = config["TRANSCRIBER"]["lm_weight"],
							word_score = config["TRANSCRIBER"]["word_score"],
							beam_size = config["TRANSCRIBER"]["beam_size"])

	audioList = glob.glob(os.path.join(config["TRANSCRIBER"]["wav_folder"], '*.wav'))
	print(audioList)

	hypos = transcriber.transcribe(audioList)
	print(hypos)

main()