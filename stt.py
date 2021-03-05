import ast
import logging
import math
import os
import sys
import editdistance
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from shutil import copy2
import uuid
import soundfile
import time

def add_asr_eval_argument(parser, lm_type, lm_model, lm_weight, word_score, lexicon, beam_size):
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary output units",
    )
    try:
        parser.add_argument(
            "--lm-weight",
            "--lm_weight",
            type=float,
            default=lm_weight,
            help="weight for lm while interpolating with neural score",
        )
    except:
        pass
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    
#     parser.add_argument(
#         "--w2l-decoder",
#         choices=["viterbi", "kenlm", "fairseqlm"],
#         help="use a w2l decoder",
#     )

    parser.add_argument("--w2l-decoder",default=lm_type,
                        help="use a w2l decoder",)
    parser.add_argument("--lexicon", help="lexicon for w2l decoder", default=lexicon)
    parser.add_argument("--unit-lm", action="store_true", help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model", help="lm model for w2l decoder", default=lm_model)
    parser.add_argument("--beam-threshold", type=float, default=beam_size)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--word-score", type=float, default=word_score)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    return parser

def check_args(args):
    # assert args.path is not None, "--path required for generation!"
    # assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def get_dataset_itr(args, task, models):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
    args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id
):
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())

        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(hyp_pieces, speaker, id),
                file=res_files["hypo.units"],
            )
            print(
                "{} ({}-{})".format(hyp_words, speaker, id),
                file=res_files["hypo.words"],
            )

        tgt_pieces = tgt_dict.string(target_tokens)
        tgt_words = post_process(tgt_pieces, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(tgt_pieces, speaker, id),
                file=res_files["ref.units"],
            )
            print(
                "{} ({}-{})".format(tgt_words, speaker, id), file=res_files["ref.words"]
            )
            # only score top hypothesis

        hyp_words = hyp_words.split()
        tgt_words = tgt_words.split()
        #return editdistance.eval(hyp_words, tgt_words), len(tgt_words)
        return 0, len(tgt_words)


def prepare_result_files(args):
    def get_res_file(file_prefix):
        if args.num_shards > 1:
            #file_prefix = f"{args.shard_id}_{file_prefix}"
            pass
        path = os.path.join(
            args.results_path,
            "{}-{}-{}.txt".format(
                file_prefix, os.path.basename(args.path), args.gen_subset
            ),
        )
        return open(path, "w", buffering=1)

    if not args.results_path:
        return None

    return {
        "hypo.words": get_res_file("hypo.word"),
        "hypo.units": get_res_file("hypo.units"),
        "ref.words": get_res_file("ref.word"),
        "ref.units": get_res_file("ref.units"),
    }


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

class ExistingEmissionsDecoder(object):
    def __init__(self, decoder, emissions):
        self.decoder = decoder
        self.emissions = emissions

    def generate(self, models, sample, **unused):
        ids = sample["id"].cpu().numpy()
        try:
            emissions = np.stack(self.emissions[ids])
        except:
            print([x.shape for x in self.emissions[ids]])
            raise Exception("invalid sizes")
        emissions = torch.from_numpy(emissions)
        return self.decoder.decode(emissions)
    
    
import random, struct, wave

def generate_random_wav(wav_path,sr = 16000):
    noise_output = wave.open(wav_path, 'w')
    noise_output.setparams((1, 2, sr, 0, 'NONE', 'not compressed'))

    for i in range(0, sr*3):
        value = random.randint(-32767, 32767)
        packed_value = struct.pack('h', value)
        noise_output.writeframes(packed_value)

    noise_output.close()

sys.argv.append('/mnt/disks2/data')

class Transcriber:
    def __init__(self, pretrain_model, finetune_model, dictionary, lm_type, lm_lexicon, lm_model,
                 lm_weight = 1.51, word_score = 2.57, beam_size = 100,
                 temp_path = 'temp'):
        
        '''
        w2vec    : path to wav2vec model
        lexicon  : path to dictionary file
        lm_model : path to language model
        lmweight  : how much language model affect the result, the higher the more important
        wordscore : weight score for group of letter forming a word
        beamsize  : number of path for decoding, the higher the better but slower
        temp_path : directory for storing temporary files during processing
        '''
        
        parser = options.get_generation_parser()
        parser = add_asr_eval_argument(parser, lm_type, lm_model, lm_weight, word_score, lm_lexicon, beam_size)
        args = options.parse_args_and_arch(parser)
        args.task = 'audio_pretraining'
        args.path = finetune_model
        args.nbest = 1
        args.criterion = 'ctc'
        args.labels = 'ltr'
        args.post_process = 'letter'
        args.max_tokens = 4000000
        args.w2vec_dict = dictionary
        self.args = args
        self.models = None
        self.saved_cfg = None
        self.generator = None
        self.state = None
        self.temp_path = os.path.abspath(temp_path)
        self.pretrain_model = os.path.abspath(pretrain_model)
        self.beam_size = beam_size
        
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        
        # Transcribe a test sample
        sample_audio_path = os.path.join(self.temp_path,'noise.wav')
        generate_random_wav(sample_audio_path,16000)
        self.transcribe([sample_audio_path])
        os.remove(sample_audio_path)
        print("Loading completed !")
        
    def transcribe(self,wav_files):
        process_dir = uuid.uuid1().hex
        process_dir = os.path.join(self.temp_path, process_dir)
        os.makedirs(process_dir)
        self.args.data=process_dir
        self.args.gen_subset='test'
        self.args.results_path=process_dir
        copy2(self.args.w2vec_dict,process_dir)
        
        test_words = os.path.join(process_dir,'test.wrd')
        test_letters = os.path.join(process_dir,'test.ltr')
        test_map = os.path.join(process_dir,'test.tsv')
        
        paths = [os.path.abspath(d) for d in wav_files]
        for i in range(0,len(paths)):
            audio_info = soundfile.info(paths[i])
            frames = audio_info.frames
            paths[i] = paths[i] + '\t' + str(frames)
        
        words = ['THIS IS A SAMPLE'] * len(paths)
        letters = [d.replace(' ','|') for d in words]
        letters = [' '.join(list(d)) + ' |' for d in letters]
        
        with open(test_words,'w') as f:
            f.write('\n'.join(words))
        
        with open(test_letters,'w') as f:
            f.write('\n'.join(letters))
        
        with open(test_map,'w') as f:
            f.write('\n')
            f.write('\n'.join(paths))

        args = self.args
        
        if args.max_tokens is None and args.batch_size is None:
            args.max_tokens = 4000000

        use_cuda = torch.cuda.is_available() and not args.cpu
        task = tasks.setup_task(args)
        
        if self.state is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(args.path, None)
            state['cfg']['model']['w2v_path'] = self.pretrain_model
            state['cfg']['generation']['beam'] = self.beam_size
            self.state = state
        else:
            state = self.state

        if self.models is None:
            models, saved_cfg = checkpoint_utils.load_model_ensemble(
                utils.split_paths(args.path),
                arg_overrides=ast.literal_eval(args.model_overrides),
                task=task,
                suffix=args.checkpoint_suffix,
                strict=(args.checkpoint_shard_count == 1),
                num_shards=args.checkpoint_shard_count,
                state=state,)
            self.models, self.saved_cfg = models, saved_cfg
        else:
            models, saved_cfg = self.models, self.saved_cfg
        
        optimize_models(args, use_cuda, models)
        task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)
        
        # Set dictionary
        tgt_dict = task.target_dictionary

        # hack to pass transitions to W2lDecoder
        if args.criterion == "asg_loss":
            raise NotImplementedError("asg_loss is currently not supported")
            # trans = criterions[0].asg.trans.data
            # args.asg_transitions = torch.flatten(trans).tolist()

        # Load dataset (possibly sharded)
        itr = get_dataset_itr(args, task, models)

        # Initialize generator
        gen_timer = StopwatchMeter()

        def build_generator(args):
            w2l_decoder = getattr(args, "w2l_decoder", None)
            if w2l_decoder == "viterbi":
                from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

                return W2lViterbiDecoder(args, task.target_dictionary)
            elif w2l_decoder == "kenlm":
                from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

                return W2lKenLMDecoder(args, task.target_dictionary)
            elif w2l_decoder == "fairseqlm":
                from examples.speech_recognition.w2l_decoder import W2lFairseqLMDecoder

                return W2lFairseqLMDecoder(args, task.target_dictionary)
            else:
                print(
                    "only wav2letter decoders with (viterbi, kenlm, fairseqlm) options are supported at the moment"
                )

        # please do not touch this unless you test both generate.py and infer.py with audio_pretraining task
        if self.generator is None:
            generator = build_generator(args)
        else:
            generator = self.generator

        if args.load_emissions:
            generator = ExistingEmissionsDecoder(
                generator, np.load(args.load_emissions, allow_pickle=True)
            )

        num_sentences = 0

        if args.results_path is not None and not os.path.exists(args.results_path):
            os.makedirs(args.results_path)

        max_source_pos = (
            utils.resolve_max_positions(
                task.max_positions(), *[model.max_positions() for model in models]
            ),
        )

        if max_source_pos is not None:
            max_source_pos = max_source_pos[0]
            if max_source_pos is not None:
                max_source_pos = max_source_pos[0] - 1

        if args.dump_emissions:
            emissions = {}
        if args.dump_features:
            features = {}
            models[0].bert.proj = None
        else:
            res_files = prepare_result_files(args)
        errs_t = 0
        lengths_t = 0
        with progress_bar.build_progress_bar(args, itr) as t:
            wps_meter = TimeMeter()
            for sample in t:
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if "net_input" not in sample:
                    continue

                prefix_tokens = None
                if args.prefix_size > 0:
                    prefix_tokens = sample["target"][:, : args.prefix_size]

                gen_timer.start()
                if args.dump_emissions:
                    with torch.no_grad():
                        encoder_out = models[0](**sample["net_input"])
                        emm = models[0].get_normalized_probs(encoder_out, log_probs=True)
                        emm = emm.transpose(0, 1).cpu().numpy()
                        for i, id in enumerate(sample["id"]):
                            emissions[id.item()] = emm[i]
                        continue
                elif args.dump_features:
                    with torch.no_grad():
                        encoder_out = models[0](**sample["net_input"])
                        feat = encoder_out["encoder_out"].transpose(0, 1).cpu().numpy()
                        for i, id in enumerate(sample["id"]):
                            padding = (
                                encoder_out["encoder_padding_mask"][i].cpu().numpy()
                                if encoder_out["encoder_padding_mask"] is not None
                                else None
                            )
                            features[id.item()] = (feat[i], padding)
                        continue
                hypos = task.inference_step(generator, models, sample, prefix_tokens)
                num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
                gen_timer.stop(num_generated_tokens)

                for i, sample_id in enumerate(sample["id"].tolist()):
                    speaker = None
                    # id = task.dataset(args.gen_subset).ids[int(sample_id)]
                    id = sample_id
                    toks = (
                        sample["target"][i, :]
                        if "target_label" not in sample
                        else sample["target_label"][i, :]
                    )
                    target_tokens = utils.strip_pad(toks, tgt_dict.pad()).int().cpu()
                    # Process top predictions
                    errs, length = process_predictions(
                        args,
                        hypos[i],
                        None,
                        tgt_dict,
                        target_tokens,
                        res_files,
                        speaker,
                        id,
                    )
                    errs_t += errs
                    lengths_t += length

                wps_meter.update(num_generated_tokens)
                t.log({"wps": round(wps_meter.avg)})
                num_sentences += (
                    sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
                )

        wer = None
        if args.dump_emissions:
            emm_arr = []
            for i in range(len(emissions)):
                emm_arr.append(emissions[i])
            np.save(args.dump_emissions, emm_arr)
        elif args.dump_features:
            feat_arr = []
            for i in range(len(features)):
                feat_arr.append(features[i])
            np.save(args.dump_features, feat_arr)
        else:
            if lengths_t > 0:
                wer = errs_t * 100.0 / lengths_t
        
        hypo_file = [file for file in os.listdir(process_dir) if 'hypo.word' in file][0]
        hypo_file = os.path.join(process_dir,hypo_file)
        
        with open(hypo_file) as f:
            hypos = f.read().splitlines()
            
        for i in range(0,len(hypos)):
            words = ' '.join(hypos[i].split()[:-1])
            idx_  = hypos[i].split()[-1].split('-')[1][:-1]
            hypos[i] = (words,int(idx_))
            
        hypos = sorted(hypos, key = lambda x : x[1])
        hypos = [h[0] for h in hypos] 
        
        os.system('rm -rf ' + process_dir)
        return hypos
