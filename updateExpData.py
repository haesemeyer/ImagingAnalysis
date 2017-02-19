# tries to update an experimental data pickle file to contain the most recent data form

from mh_2P import TailData, UiGetFile, SOORepeatExperiment, HeatPulseExperiment, SLHRepeatExperiment
from mh_2P import TailDataDict, RepeatExperiment
import numpy as np
import pickle
from os import path, makedirs
from datetime import datetime


def recomputeVigor(data, tc):
    """

    Args:
        data (RepeatExperiment): The data structure to update
        tc: The (new) calcium indicator timeconstant to use for the update
    """
    n_frames = (data.preFrames + data.stimFrames + data.postFrames) * data.nRepeats
    n_seconds = n_frames / data.frameRate
    tdd = TailDataDict(tc)
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
    return vigor.astype(np.float32)


def update_experiment_caKernel(data, tc):
    """

        Args:
            data (RepeatExperiment): The data structure to update
            tc: The (new) calcium indicator timeconstant to use for the update
    """
    if type(data) is not SLHRepeatExperiment:
        raise ValueError("Can only update experiments of type SLHRepeatExperiment")
    new_vigor = recomputeVigor(data, tc)
    activity = data.RawData
    new_data = SLHRepeatExperiment(activity, new_vigor, data.preFrames, data.stimFrames, data.postFrames,
                                   data.nRepeats, tc,
                                   nHangoverFrames=data.nHangoverFrames, frameRate=data.frameRate)
    new_data.graph_info = data.graph_info
    new_data.original_time_per_frame = data.original_time_per_frame
    return new_data

if __name__ == "__main__":
    new_timeconstant = 3.0
    print("Load data files")
    dfnames = UiGetFile([('Experiment data', '.pickle')], True)
    folder_name = "/backup_" + str(datetime.today().toordinal())
    for name in dfnames:
        contFolder = path.dirname(name)
        bk_filename = path.split(name)[1]
        bkup_dir = contFolder + folder_name
        if not path.exists(bkup_dir):
            makedirs(bkup_dir)
        backup_file = bkup_dir + '/' + bk_filename
        # load data
        f = open(name, 'rb')
        d = pickle.load(f)
        f.close()
        # make backup copy of data in case we screw things up..
        f = open(backup_file, 'wb')
        pickle.dump(d, f)
        f.close()
        # update data structure
        d = update_experiment_caKernel(d, new_timeconstant)
        # save data back to original filename
        f = open(name, 'wb')
        pickle.dump(d, f)
        f.close()
