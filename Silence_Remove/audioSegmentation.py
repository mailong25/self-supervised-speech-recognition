from __future__ import absolute_import
from . import audioBasicIO
from . import audioFeatureExtraction as aF
from . import audioTrainTest as aT
import numpy
import matplotlib.pyplot as plt

def silenceRemoval(x, fs, st_win, st_step, smoothWindow=0.5, weight=0.5, plot=False):
    '''
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - fs:               sampling freq
         - st_win, st_step:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - weight:           (optinal) weight factor (0 < weight < 1) the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - seg_limits:    list of segment limits in seconds (e.g [[0.1, 0.9], [1.4, 3.0]] means that
                    the resulting segments are (0.1 - 0.9) seconds and (1.4, 3.0) seconds
    '''

    if weight >= 1:
        weight = 0.99
    if weight <= 0:
        weight = 0.01

    # Step 1: feature extraction
    x = audioBasicIO.stereo2mono(x)
    st_feats, _ = aF.stFeatureExtraction(x, fs, st_win * fs, 
                                                  st_step * fs)

    # Step 2: train binary svm classifier of low vs high energy frames
    # keep only the energy short-term sequence (2nd feature)
    st_energy = st_feats[1, :]
    en = numpy.sort(st_energy)
    # number of 10% of the total short-term windows
    l1 = int(len(en) / 10)
    # compute "lower" 10% energy threshold
    t1 = numpy.mean(en[0:l1]) + 0.000000000000001
    # compute "higher" 10% energy threshold
    t2 = numpy.mean(en[-l1:-1]) + 0.000000000000001
    # get all features that correspond to low energy
    class1 = st_feats[:, numpy.where(st_energy <= t1)[0]]
    # get all features that correspond to high energy
    class2 = st_feats[:, numpy.where(st_energy >= t2)[0]]
    # form the binary classification task and ...
    faets_s = [class1.T, class2.T]
    # normalize and train the respective svm probabilistic model
    # (ONSET vs SILENCE)
    [faets_s_norm, means_s, stds_s] = aT.normalizeFeatures(faets_s)
    svm = aT.trainSVM(faets_s_norm, 1.0)

    # Step 3: compute onset probability based on the trained svm
    prob_on_set = []
    for i in range(st_feats.shape[1]):
        # for each frame
        cur_fv = (st_feats[:, i] - means_s) / stds_s
        # get svm probability (that it belongs to the ONSET class)
        prob_on_set.append(svm.predict_proba(cur_fv.reshape(1,-1))[0][1])
    prob_on_set = numpy.array(prob_on_set)
    # smooth probability:
    prob_on_set = smoothMovingAvg(prob_on_set, smoothWindow / st_step)

    # Step 4A: detect onset frame indices:
    prog_on_set_sort = numpy.sort(prob_on_set)
    # find probability Threshold as a weighted average
    # of top 10% and lower 10% of the values
    Nt = int(prog_on_set_sort.shape[0] / 10)
    T = (numpy.mean((1 - weight) * prog_on_set_sort[0:Nt]) +
         weight * numpy.mean(prog_on_set_sort[-Nt::]))

    max_idx = numpy.where(prob_on_set > T)[0]
    # get the indices of the frames that satisfy the thresholding
    i = 0
    time_clusters = []
    seg_limits = []

    # Step 4B: group frame indices to onset segments
    while i < len(max_idx):
        # for each of the detected onset indices
        cur_cluster = [max_idx[i]]
        if i == len(max_idx)-1:
            break
        while max_idx[i+1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_idx[i+1])
            i += 1
            if i == len(max_idx)-1:
                break
        i += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step,
                           cur_cluster[-1] * st_step])

    # Step 5: Post process: remove very small segments:
    min_dur = 0.3
    seg_limits_2 = []
    for s in seg_limits:
        if s[1] - s[0] > min_dur:
            seg_limits_2.append(s)
    seg_limits = seg_limits_2

    if plot:
        timeX = numpy.arange(0, x.shape[0] / float(fs), 1.0 / fs)
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(timeX, x)
        for s in seg_limits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.subplot(2, 1, 2)
        plt.plot(numpy.arange(0, prob_on_set.shape[0] * st_step, st_step), 
                 prob_on_set)
        plt.title('Signal')
        for s in seg_limits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.title('svm Probability')
        plt.show()

    return seg_limits

def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = numpy.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1],
                 inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
    w = numpy.ones(windowLen, 'd')
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[windowLen:-windowLen+1]

