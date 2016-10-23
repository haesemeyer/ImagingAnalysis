from mh_2P import OpenStack, TailData, UiGetFile, NucGraph, CorrelationGraph, SOORepeatExperiment, SLHRepeatExperiment
import numpy as np
import nimfa
from scipy.signal import savgol_filter

import matplotlib.pyplot as pl
import seaborn as sns

import pickle


def dff(fluomat):
    f0 = np.median(fluomat[:, :144], 1, keepdims=True)
    return(fluomat-f0)/f0


def NonNegMatFact(rawData, frameRate, nComponents, beta=5e-4):
    # filter data with savitzky golay filter - polynomial order 3
    # win_len = int(2 * frameRate)
    # if win_len % 2 == 0:
    #     win_len += 1
    # fil_data = savgol_filter(rawData, win_len, 3, axis=1)
    fil_data = rawData.copy()
    # normalize data (to prevent outlier effect normalize by 95th percentile not max)
    fil_data -= np.min(fil_data, 1, keepdims=True)  # we need to ensure that no negative values present so true min used
    max99 = np.percentile(fil_data, 99, 1, keepdims=True)
    fil_data /= max99
    snmf = nimfa.Nmf(fil_data, seed="random_vcol", rank=nComponents, max_iter=100, n_run=30,
                     update='divergence', objective='div')
    return snmf, snmf(), fil_data


def n_r2_above_thresh(corr_mat, r2_thresh):
    """
    For a given trace correlation matrix computes for each trace
    how many other traces correlate with it above a given r-squared
    threshold
    Args:
        corr_mat: The correlation matrix
        r2_thresh: The threshold on the R2

    Returns:
        The number of non-self traces that correlate above the given threshold

    """
    return np.sum(corr_mat**2 > r2_thresh, 1) - 1


def n_exp_r2_above_thresh(corr_mat, r2_thresh, exp_ids):
    """
    For each trace in the corrleation matrix computes how many
    other experiments contain at least one trace that correlates
    above the given r-square threshold
    Args:
        corr_mat: The correlation matrix
        r2_thresh: The threshold in the R2
        exp_ids: The experiment id's for each trace

    Returns:
        The number of non-self experiments that have a correlating trace above the threshold
    """
    corr_above = corr_mat**2 > r2_thresh
    return np.array([np.unique(exp_ids[above]).size-1 for above in corr_above])


def MakeCorrelationGraphStack(experiment_data, corr_red, corr_green, corr_blue, cutOff=0.5):
    def ZPIndex(fname):
        try:
            ix = int(fname[-8:-6])
        except ValueError:
            ix = int(fname[-7:-6])
        return ix

    stack_filenames = set([info[0] for info in experiment_data.graph_info])
    stack_filenames = sorted(stack_filenames, key=ZPIndex)
    z_stack = np.zeros((len(stack_filenames), 512, 512, 3))
    for i, sfn in enumerate(stack_filenames):
        # load realigned stack
        stack = np.load(sfn[:-4] + "_stack.npy").astype('float')
        sum_stack = np.sum(stack, 0)
        sum_stack /= np.percentile(sum_stack, 99)
        sum_stack[sum_stack > 0.8] = 0.8
        projection = np.zeros((sum_stack.shape[0], sum_stack.shape[1], 3), dtype=float)
        projection[:, :, 0] = projection[:, :, 1] = projection[:, :, 2] = sum_stack
        for j, gi in enumerate(experiment_data.graph_info):
            if gi[0] != sfn:
                continue
            # this graph-info comes from the same plane, check if we corelate
            if corr_red[j] > cutOff:
                for v in gi[1]:
                    projection[v[0], v[1], 0] = 1
            if corr_green[j] > cutOff:
                for v in gi[1]:
                    projection[v[0], v[1], 1] = 1
            if corr_blue[j] > cutOff:
                for v in gi[1]:
                    projection[v[0], v[1], 2] = 1
        z_stack[i, :, :, :] = projection
    return z_stack


def MakeMaskStack(experiment_data, col_red, col_green, col_blue, cutOff=0.0, scaleMax=1.0):
    def ZPIndex(fname):
        try:
            ix = int(fname[-8:-6])
        except ValueError:
            ix = int(fname[-7:-6])
        return ix

    # rescale channels
    col_red[col_red < cutOff] = 0
    col_red /= scaleMax
    col_red[col_red > 1] = 1
    col_green[col_green < cutOff] = 0
    col_green /= scaleMax
    col_green[col_green > 1] = 1
    col_blue[col_blue < cutOff] = 0
    col_blue /= scaleMax
    col_blue[col_blue > 1] = 1
    stack_filenames = set([info[0] for info in experiment_data.graph_info])
    stack_filenames = sorted(stack_filenames, key=ZPIndex)
    z_stack = np.zeros((len(stack_filenames), 512, 512, 3))
    for i, sfn in enumerate(stack_filenames):
        projection = np.zeros((512, 512, 3), dtype=float)
        for j, gi in enumerate(experiment_data.graph_info):
            if gi[0] != sfn:
                continue
            # this graph-info comes from the same plane color according to our channel information
            for v in gi[1]:
                projection[v[0], v[1], 0] = col_red[j]
                projection[v[0], v[1], 1] = col_green[j]
                projection[v[0], v[1], 2] = col_blue[j]
        z_stack[i, :, :, :] = projection
    return z_stack


def SaveProjectionStack(stack):
    from PIL import Image
    for i in range(stack.shape[0]):
        im_roi = Image.fromarray((stack[i, :, :, :] * 255).astype(np.uint8))
        im_roi.save('Z_' + str(i).zfill(3) + '.png')


def MakeAndSaveMajorTypeStack(experiment_data):
    pot = np.load('pot_transOn.npy')
    pot_tOn = np.zeros_like(pot)
    pot_tOn[:pot.size // 2] = pot[:pot.size//2] + pot[pot.size//2:]
    pot_tOn[pot.size // 2:] = pot[:pot.size//2] + pot[pot.size//2:]
    # limit cells to those that show at least 1std deviation of pre-activity in their activity modulation
    stim_fluct = experiment_data.computeStimulusEffect(0)[0].flatten()
    no_act = stim_fluct <= 1
    c_on = np.array([np.corrcoef(experiment_data.stimOn, trace)[0, 1] for trace in experiment_data.RawData])
    c_on[no_act] = 0
    c_off = np.array([np.corrcoef(experiment_data.stimOff, trace)[0, 1] for trace in experiment_data.RawData])
    c_off[no_act] = 0
    # c_ton = np.array([np.corrcoef(pot_tOn, trace)[0, 1] for trace in experiment_data.RawData])
    # c_ton[no_act] = 0
    zstack = MakeCorrelationGraphStack(experiment_data, c_on, c_on, c_off)
    SaveProjectionStack(zstack)


def MakeAndSaveRegressionStack(experiment_data):
    global orthonormals
    r2_vals = np.zeros(experiment_data.RawData.shape[0])
    for i, row in enumerate(experiment_data.RawData):
        if np.any(np.isnan(row)):
            continue
        lreg = LinearRegression()
        lreg.fit(orthonormals, row)
        r2_vals[i] = lreg.score(orthonormals, row)
    zstack = MakeCorrelationGraphStack(experiment_data, r2_vals, np.zeros_like(r2_vals), np.zeros_like(r2_vals),
                                       cutOff=0.6)
    SaveProjectionStack(zstack)


def MakeAndSaveROIStack(experiment_data):
    """
    Creates an ROI only (i.e. no gcamp background) stack of sensory driven units scaled by R2 in red
    and motor driven units scaled by R2 in blue
    """
    global orthonormals
    r2_sensory = np.zeros(experiment_data.RawData.shape[0])
    r2_motor = np.zeros_like(r2_sensory)
    for i, row in enumerate(experiment_data.RawData):
        if np.any(np.isnan(row)):
            continue
        lreg = LinearRegression()
        lreg.fit(orthonormals, row)
        r2_sensory[i] = lreg.score(orthonormals, row)
        if np.any(np.isnan(experiment_data.Vigor[i, :])):
            continue
        r2_motor[i] = np.corrcoef(experiment_data.Vigor[i, :], row)[0, 1]**2
    zstack = MakeMaskStack(experiment_data, r2_sensory, np.zeros_like(r2_sensory), r2_motor, 0.5, 1.0)
    SaveProjectionStack(zstack)


def expVigor(expData):
    done = dict()
    vigs = []
    for i in range(expData.Vigor.shape[0]):
        if expData.graph_info[i][0] in done:
            continue
        done[expData.graph_info[i][0]] = True
        vigs.append(expData.Vigor[i, :])
    return np.vstack(vigs)

if __name__ == "__main__":
    print("Load data files")
    exp_data = []
    dfnames = UiGetFile([('Experiment data', '.pickle')], True)
    for name in dfnames:
        f = open(name, 'rb')
        d = pickle.load(f)
        exp_data.append(d)
    is_pot_stim = np.array([], dtype=np.bool)  # for each unit whether it is potentially a stimulus driven unit
    exp_id = np.array([])  # for each unit the experiment which it came from
    stim_phase = np.array([])  # for each unit the phase at stimulus frequency during the sine-presentation
    for i, data in enumerate(exp_data):
        m_corr = data.motorCorrelation(0)[0].flatten()
        stim_fluct = data.computeStimulusEffect(0)[0].flatten()
        exp_id = np.r_[exp_id, np.full(m_corr.size, i, np.int32)]
        # ips = stim_fluct > (m_sh_sid + 2 * std_sh_sid)
        # ips = np.logical_and(ips, stim_fluct >= 1)
        ips = stim_fluct >= 1
        ips = np.logical_and(ips, m_corr < 0.4)
        is_pot_stim = np.r_[is_pot_stim, ips]
        if i == 0:
            all_activity = data.RawData
        else:
            all_activity = np.r_[all_activity, data.RawData]
        p = data.computeFourierMetrics()[4]
        stim_phase = np.r_[stim_phase, p]
    # compute correlation matrix of all time-series data
    corr_mat = np.corrcoef(all_activity[is_pot_stim, :])
    # get all cells that have at least 10 other cells with a timeseries R2>0.5 and are spread
    # across at least 2 experiments
    exp_g_1 = n_exp_r2_above_thresh(corr_mat, 0.5, exp_id[is_pot_stim]) > 2
    c_g_9 = n_r2_above_thresh(corr_mat, 0.5) > 19
    to_analyze = np.logical_and(exp_g_1, c_g_9)
    analysis_data = all_activity[is_pot_stim, :][to_analyze, :]

    # analyze per-experiment swim vigors
    all_expVigors = np.vstack([np.mean(expVigor(data), 0) for data in exp_data])
    all_expVigors = all_expVigors / np.mean(all_expVigors, 1, keepdims=True)

    # TODO: The following is experiment specific - needs cleanup!
    avg_analysis_data = np.zeros((analysis_data.shape[0], analysis_data.shape[1] // 3))
    avg_analysis_data += analysis_data[:, :825]
    avg_analysis_data += analysis_data[:, 825:825*2]
    avg_analysis_data += analysis_data[:, 825*2:]
    norm_data = avg_analysis_data/np.percentile(avg_analysis_data, 20, 1, keepdims=True)  # F/F0
    # Note: currently we anyway normalize in NonNegMatFact so normalization above kinda useless
    nnmf, fit, fildata = nnmf, fit, fildata = NonNegMatFact(norm_data, 5, 5)
    W = np.array(nnmf.W)  # cells / components weight matrix
