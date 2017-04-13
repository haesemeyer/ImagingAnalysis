import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from mh_2P import DetailCharExperiment, MotorContainer, UiGetFile
from motorPredicates import high_bias_bouts, unbiased_bouts
import pickle
from typing import List
from scipy.stats import wilcoxon


def dff(ts):
    f0 = np.percentile(ts, 10, axis=1, keepdims=True)
    f0[f0 < 0.1] = 0.1
    return (ts-f0)/f0


if __name__ == "__main__":
    sns.reset_orig()
    # Load data files
    print("Load data files", flush=True)
    exp_data = []  # type: List[DetailCharExperiment]
    dfnames = UiGetFile([('Experiment data', '.pickle')], True)
    for name in dfnames:
        f = open(name, 'rb')
        d = pickle.load(f)
        exp_data.append(d)
    # generate our (sourceFile, ca-frame-time) tuple list for the motor containers
    source_files = []
    var_scores = np.array([])
    all_activity = np.array([])
    for i, data in enumerate(exp_data):
        source_files += [(gi[0], data.original_time_per_frame) for gi in data.graph_info]
        var_scores = np.r_[var_scores, data.variance_score()]
        if i == 0:
            # NOTE: RawData field is 64 bit float - convert to 32 when storing in all_activity
            all_activity = data.RawData.astype(np.float32)
        else:
            all_activity = np.r_[all_activity, data.RawData.astype(np.float32)]
    var_scores[np.isnan(var_scores)] = 0
    # create motor containers
    i_time = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    tc = exp_data[0].caTimeConstant
    mc_all = MotorContainer(source_files, i_time, tc)
    mc_high_bias = MotorContainer(source_files, i_time, tc, high_bias_bouts, tdd=mc_all.tdd)
    mc_low_bias = MotorContainer(source_files, i_time, tc, unbiased_bouts, tdd=mc_all.tdd)
    mc_hb_raw = MotorContainer(source_files, i_time, 0, high_bias_bouts, tdd=mc_all.tdd)
    mc_lb_raw = MotorContainer(source_files, i_time, 0, unbiased_bouts, tdd=mc_all.tdd)
    mc = [mc_all, mc_high_bias, mc_low_bias]

    # plot average probability of motor events
    fig, ax = pl.subplots()
    ax.plot(np.mean(exp_data[0].repeat_align(mc_hb_raw.avg_motor_output), 1).ravel(), label="Flicks")
    ax.plot(np.mean(exp_data[0].repeat_align(mc_lb_raw.avg_motor_output), 1).ravel(), label="Swims")
    ax.legend()
    sns.despine(fig, ax)

    # use motor types to sort out motor units
    mc_type_corrs = np.zeros((all_activity.shape[0], len(mc)))
    for i, act in enumerate(all_activity):
        for j, m in enumerate(mc):
            corr = np.corrcoef(act, m[i])[0, 1]
            if np.isnan(corr):
                corr = 0
            mc_type_corrs[i, j] = corr
    is_pot_stim = np.logical_and(var_scores > 0.05, np.max(mc_type_corrs, 1) < 0.4)

    # find units that show significant activation during 1500mW step and sine period
    p_stp = []
    p_sin = []
    for data in exp_data:
        pre, stim = data.activations(*data.get_lstep_pre_stim_ix())
        p_stp += [wilcoxon(p, s)[1] for p, s in zip(pre, stim)]
        pre, stim = data.activations(*data.get_sine_pre_stim_ix())
        p_sin += [wilcoxon(p, s)[1] for p, s in zip(pre, stim)]
    p_stp = np.array(p_stp)
    p_sin = np.array(p_sin)
    # mark units that show significant change in both periods
    sig_act = np.logical_and(p_stp < 0.01, p_sin < 0.01)
    stim_units = np.logical_and(sig_act, is_pot_stim)
