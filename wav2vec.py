import argparse
import glob
import os
from shutil import copy
import h5py
import soundfile as sf
import numpy as np
import torch
from torch import nn
import tqdm

from fairseq.models.wav2vec import Wav2VecModel

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """
    wav, sr = sf.read(fname)
    assert sr == 16e3
    return wav, 16e3

class PretrainedWav2VecModel(nn.Module):

    def __init__(self, fname):
        super().__init__()

        checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        self.args = checkpoint["args"]
        model = Wav2VecModel.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            z = self.model.feature_extractor(x)
            if isinstance(z, tuple):
                z = z[0]
            c = self.model.feature_aggregator(z)
        return z, c


class Prediction():
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname)
        if torch.cuda.is_available():
            self.model = self.model.cuda(0)

    def __call__(self, x):
        x = torch.from_numpy(x).float()
        if torch.cuda.is_available():
            x = x.cuda(0)
        with torch.no_grad():
            z, c = self.model(x.unsqueeze(0))

        return z.squeeze(0).cpu().numpy(), c.squeeze(0).cpu().numpy()

class H5Writer():
    """ Write features as hdf5 file in wav2letter++ compatible format """

    def __init__(self, fname):
        self.fname = fname
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)

    def write(self, data):
        channel, T = data.shape

        with h5py.File(self.fname, "w") as out_ds:
            data = data.T.flatten()
            out_ds["features"] = data
            out_ds["info"] = np.array([16e3 // 160, T, channel])


class EmbeddingDatasetWriter(object):
    """ Given a model and a wav2letter++ dataset, pre-compute and store embeddings
    Args:
        input_root, str :
            Path to the wav2letter++ dataset
        output_root, str :
            Desired output directory. Will be created if non-existent
    """

    def __init__(self, input_root, output_root,
                 loaded_model,
                 extension="wav",
                 verbose=False,
                 use_feat=False):
        
        self.model = loaded_model
        self.input_root = input_root
        self.output_root = output_root
        self.verbose = verbose
        self.extension = extension
        self.use_feat = use_feat

        assert os.path.exists(self.input_path), \
            "Input path '{}' does not exist".format(self.input_path)

    def _progress(self, iterable, **kwargs):
        if self.verbose:
            return tqdm.tqdm(iterable, **kwargs)
        return iterable

    def require_output_path(self, fname=None):
        path = self.get_output_path(fname)
        os.makedirs(path, exist_ok=True)

    @property
    def input_path(self):
        return self.get_input_path()

    @property
    def output_path(self):
        return self.get_output_path()

    def get_input_path(self, fname=None):
        if fname is None:
            return os.path.join(self.input_root)
        return os.path.join(self.get_input_path(), fname)

    def get_output_path(self, fname=None):
        if fname is None:
            return os.path.join(self.output_root)
        return os.path.join(self.get_output_path(), fname)

    def copy_labels(self):
        self.require_output_path()

        labels = list(filter(lambda x: self.extension not in x, glob.glob(self.get_input_path("*"))))
        for fname in tqdm.tqdm(labels):
            copy(fname, self.output_path)

    @property
    def input_fnames(self):
        return sorted(glob.glob(self.get_input_path("*.{}".format(self.extension))))

    def __len__(self):
        return len(self.input_fnames)

    def write_features(self):

        paths = self.input_fnames

        fnames_context = map(lambda x: os.path.join(self.output_path, x.replace("." + self.extension, ".h5context")), \
                             map(os.path.basename, paths))

        for name, target_fname in self._progress(zip(paths, fnames_context), total=len(self)):
            wav, sr = read_audio(name)
            z, c = self.model(wav)
            feat = z if self.use_feat else c
            writer = H5Writer(target_fname)
            writer.write(feat)

    def __repr__(self):
        return "EmbeddingDatasetWriter ({n_files} files)\n\tinput:\t{input_root}\n\toutput:\t{output_root})".format(
            n_files=len(self), **self.__dict__)