# file to analyze 2-photon imaging data from sine-on-off experiments with repeat presentation
# unit graphs should have been constructed and saved before using analyzeStack.py script

from mh_2P import OpenStack, TailData, UiGetFile, NucGraph, CorrelationGraph, SOORepeatExperiment, HeatPulseExperiment
from mh_2P import MedianPostMatch, ZCorrectTrace, TailDataDict, SLHRepeatExperiment
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

import pickle

import sys
from scipy.signal import lfilter
sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')

import mhba_basic as mb


def tryZCorrect(graphs, eyemask=None):
    sf = graphs[0].SourceFile
    stabFile = sf.replace("_0.tif", "_stableZ_0.tif")
    try:
        stack = OpenStack(sf)
        pre = OpenStack(stabFile)
    except IOError:
        print("Could not load stableZ prestack for ", sf)
        return
    # if information is present mask out pixels on eyes
    if eyemask is not None:
        eyemask[eyemask > 0] = 1
        eyemask[eyemask <= 0] = 0
        stack = stack * eyemask
        pre = pre * eyemask
    sl = MedianPostMatch(pre, stack, interpolate=True, nIter=100)[0]
    # create slice-summed projection of pre-stack
    sum_stack = np.zeros((pre.shape[0] // 21, pre.shape[1], pre.shape[2]), dtype=np.float32)
    for i in range(sum_stack.shape[0]):
        sum_stack[i, :, :] = np.sum(pre[i * 21:(i + 1) * 21, :, :], 0)
    for g in graphs:
        g.Original = g.RawTimeseries.copy()
        g.RawTimeseries = ZCorrectTrace(sum_stack, sl, g.RawTimeseries, g.V)


if __name__ == "__main__":
    interpol_freq = 5  # the frequency to which we interpolate before analysis

    n_repeats = int(input("Please enter the number of repeats:"))  # the number of repeats performed in each plane
    s_pre = int(input("Please enter the number of pre-seconds:"))
    n_pre = s_pre * interpol_freq
    s_stim = int(input("Please enter the number of stimulus-seconds:"))
    n_stim = s_stim * interpol_freq  # the number of stimulus frames at interpol_freq
    try:
        s_post = int(input("Please enter the number of post-seconds or press enter for 75:"))
    except ValueError:
        print("Assuming default value of 75s")
        s_post = 75
    n_post = s_post * interpol_freq
    t_per_frame = float(input("Please enter the duration of each frame in seconds:"))
    ans = ""

    exp_type = int(input("Experiment type: SOORepeat [0], LoHi [1], HeatPulse[2]"))
    if exp_type < 0 or exp_type > 2:
        raise ValueError("Invalid experiment type selected")

    ans = ""
    while ans != 'n' and ans != 'y':
        ans = input("Load eye mask file? [y/n]:")
    if ans == 'n':
        eyemask = None
    else:
        mfile = UiGetFile([('Eye mask file', 'EYEMASK*')])
        eyemask = OpenStack(mfile)

    n_hangoverFrames = 0  # the number of "extra" recorded frames at experiment end - = 0 after interpolation

    n_shuffles = 200
    s_amp = 0.36  # sine amplitude relative to offset

    # load unit activity from each imaging plane, compute motor correlations
    # and repeat averaged time-trace <- note that motor correlations have to
    # be calculated on non-time averaged data
    graphFiles = UiGetFile([('PixelGraphs', '.graph')], multiple=True)
    graph_list = []

    # load all units from file
    for fname in graphFiles:
        print("Processing ", fname)
        f = open(fname, 'rb')
        graphs = pickle.load(f)
        if len(graphs) > 0:
            tryZCorrect(graphs, eyemask)
        graph_list += graphs

    # Update ca-time-constants of the graph to better reflect *nuclear* gcamp-6s
    for g in graph_list:
        # g.CaTimeConstant = 3.0
        g.CaTimeConstant = 1.76

    tdd = TailDataDict(graph_list[0].CaTimeConstant)
    # add frame bout start trace to graphs if required
    # for g in graph_list:
    #     # if hasattr(g, 'BoutStartTrace') and g.BoutStartTrace is not None:
    #     #    continue
    #     tdata = tdd[g.SourceFile]
    #     g.BoutStartTrace = tdata.FrameBoutStarts

    frame_times = np.arange(graph_list[0].RawTimeseries.size) * t_per_frame
    # endpoint=False in the following call will remove hangover frame
    interp_times = np.linspace(0, (s_pre+s_stim+s_post)*n_repeats, (n_pre+n_stim+n_post)*n_repeats, endpoint=False)

    # interpolate calcium data
    ipol = lambda y: np.interp(interp_times, frame_times, y[:frame_times.size])
    interp_data = np.vstack([ipol(g.RawTimeseries) for g in graph_list])

    # interpolate convolved bout starts (means interpolating down from original 100Hz trace)
    traces = dict()
    bstarts = []
    # for binning, we want to have one more subdivision of the times and include the endpoint - later each time-bin
    # will correspond to one timepoint in interp_times above
    i_t = np.linspace(0, (s_pre+s_stim+s_post)*n_repeats, (n_pre+n_stim+n_post)*n_repeats + 1, endpoint=True)
    for g in graph_list:
        if g.SourceFile not in traces:
            tdata = tdd[g.SourceFile]
            conv_bstarts = tdata.ConvolvedStarting
            times = tdata.frameTime
            digitized = np.digitize(times, i_t)
            t = np.array([conv_bstarts[digitized == i].sum() for i in range(1, interp_times.size+1)])
            traces[g.SourceFile] = t
        bstarts.append(traces[g.SourceFile])
    interp_starts = np.vstack(bstarts)

    if exp_type == 1:
        data = SLHRepeatExperiment(interp_data,
                                   interp_starts,
                                   n_pre, n_stim, n_post, n_repeats, graph_list[0].CaTimeConstant,
                                   nHangoverFrames=0, frameRate=interpol_freq)
    elif exp_type == 0:
        data = SOORepeatExperiment(interp_data,
                                   interp_starts,
                                   n_pre, n_stim, n_post, n_repeats, graph_list[0].CaTimeConstant,
                                   nHangoverFrames=0, frameRate=interpol_freq)
    elif exp_type == 2:
        data = HeatPulseExperiment(interp_data,
                                   interp_starts,
                                   n_pre, n_stim, n_post, n_repeats, graph_list[0].CaTimeConstant,
                                   nHangoverFrames=0, frameRate=interpol_freq, baseCurrent=0)
    else:
        raise NotImplementedError("Unknown experiment type")

    data.graph_info = [(g.SourceFile, g.V) for g in graph_list]
    data.original_time_per_frame = t_per_frame
