# tries to update an experimental data pickle file to contain the most recent data form

from mh_2P import TailData, UiGetFile, SOORepeatExperiment, HeatPulseExperiment
from mh_2P import TailDataDict, RepeatExperiment
import numpy as np
import pickle


def recomputeVigor(data):
    """

    Args:
        data (RepeatExperiment): The data structure to update
    """
    n_frames = (data.preFrames + data.stimFrames + data.postFrames) * data.nRepeats
    n_seconds = n_frames / data.frameRate
    tdd = TailDataDict()
    traces = dict()
    bstarts = []
    # for binning, we want to have one more subdivision of the times and include the endpoint - later each time-bin
    # will correspond to one timepoint in interp_times above
    i_t = np.linspace(0, n_seconds, n_frames + 1, endpoint=True)
    for info in data.graph_info:
        if info[0] not in traces:
            tdata = tdd[info[0]]
            conv_bstarts = tdata.ConvolvedStarting
            times = tdata.frameTime
            digitized = np.digitize(times, i_t)
            t = np.array([conv_bstarts[digitized == i].sum() for i in range(1, i_t.size)])
            traces[info[0]] = t
        bstarts.append(traces[info[0]])
    vigor = np.vstack(bstarts)
    assert vigor.shape[0] == data.Vigor.shape[0] and vigor.shape[1] == data.Vigor.shape[1]
    data.Vigor = vigor.astype(np.float32)


if __name__ == "__main__":
    print("Load data files")
    dfnames = UiGetFile([('Experiment data', '.pickle')], True)
    for name in dfnames:
        # load data
        f = open(name, 'rb')
        d = pickle.load(f)
        f.close()
        # make backup copy of data in case we screw things up..
        n_bkup = name + '_bkup'
        f = open(n_bkup, 'wb')
        pickle.dump(d, f)
        f.close()
        # update data structure
        recomputeVigor(d)
        # save data back to original filename
        f = open(name, 'wb')
        pickle.dump(d, f)
        f.close()
