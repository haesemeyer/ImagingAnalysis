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
    fig, ax = pl.subplots(ncols=2, sharex=True, sharey=True)
    rep_time = np.linspace(0, 135, 135*exp_data[0].frameRate, endpoint=False)
    ax[0].plot(rep_time, np.mean(exp_data[0].repeat_align(mc_hb_raw.avg_motor_output), 1).ravel())
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Response probability")
    ax[1].plot([], [])
    ax[1].plot(rep_time, np.mean(exp_data[0].repeat_align(mc_lb_raw.avg_motor_output), 1).ravel())
    ax[1].set_xlabel("Time [s]")
    ax[0].set_title("Flicks")
    ax[1].set_title("Swims")
    sns.despine(fig)
    fig.tight_layout()

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
    p_stp = []  # p-values of activation during step and sine wave and tap
    p_sin = []
    p_tap = []
    act_sign = np.array([])  # sign of activation change (ON=plus, OFF=minus)
    for data in exp_data:
        pre, stim = data.activations(*data.get_lstep_pre_stim_ix())
        p_stp += [wilcoxon(p, s)[1] for p, s in zip(pre, stim)]
        diffs = np.array([np.mean(s-p) for p, s in zip(pre, stim)])
        pre, stim = data.activations(*data.get_sine_pre_stim_ix())
        p_sin += [wilcoxon(p, s)[1] for p, s in zip(pre, stim)]
        diffs += np.array([np.mean(s-p) for p, s in zip(pre, stim)])
        act_sign = np.r_[act_sign, np.sign(diffs)]
        pre, stim = data.activations(*data.get_tap_pre_stim_ix())
        p_tap += [wilcoxon(p, s)[1] for p, s in zip(pre, stim)]
    p_stp = np.array(p_stp)
    p_sin = np.array(p_sin)
    p_tap = np.array(p_tap)
    # mark units that show significant change in both periods
    sig_act = np.logical_and(p_stp < 0.01, p_sin < 0.01)
    stim_units = np.logical_and(sig_act, is_pot_stim)
    # compute repeat-average of all stim_units
    su_avg = np.mean(data.repeat_align(all_activity[stim_units, :]), 1)

    # divide cells based on heat sign an whether they also significantly respond to taps or not
    on_tap_cells = su_avg[np.logical_and(act_sign[stim_units] > 0, p_tap[stim_units] < 0.05), :]
    on_no_tap_cells = su_avg[np.logical_and(act_sign[stim_units] > 0, p_tap[stim_units] > 0.2), :]
    off_tap_cells = su_avg[np.logical_and(act_sign[stim_units] < 0, p_tap[stim_units] < 0.05), :]
    off_no_tap_cells = su_avg[np.logical_and(act_sign[stim_units] < 0, p_tap[stim_units] > 0.2), :]

    # plot average activity of the types segregated by ON/OFF with heat and tap responses in separate plots
    fig, axes = pl.subplots(2, 2, gridspec_kw={'width_ratios': [4, 1]})
    # ON heat time
    sns.tsplot(dff(on_tap_cells[:, rep_time < 125]), rep_time[rep_time < 125], color="r", ax=axes[0][0])
    sns.tsplot(dff(on_no_tap_cells[:, rep_time < 125]), rep_time[rep_time < 125], color="orange", ax=axes[0][0])
    axes[0][0].set_xlabel("Time [s]")
    axes[0][0].set_ylabel("dF/F")
    # ON tap time
    sns.tsplot(dff(on_tap_cells[:, rep_time > 125]), rep_time[rep_time > 125], color="r", ax=axes[0][1])
    sns.tsplot(dff(on_no_tap_cells[:, rep_time > 125]), rep_time[rep_time > 125], color="orange", ax=axes[0][1])
    axes[0][1].set_xlabel("Time [s]")
    axes[0][1].set_ylabel("dF/F")
    # OFF heat time
    sns.tsplot(dff(off_tap_cells[:, rep_time < 125]), rep_time[rep_time < 125], color="b", ax=axes[1][0])
    sns.tsplot(dff(off_no_tap_cells[:, rep_time < 125]), rep_time[rep_time < 125], color="m", ax=axes[1][0])
    axes[1][0].set_xlabel("Time [s]")
    axes[1][0].set_ylabel("dF/F")
    # OFF tap time
    sns.tsplot(dff(off_tap_cells[:, rep_time > 125]), rep_time[rep_time > 125], color="b", ax=axes[1][1])
    sns.tsplot(dff(off_no_tap_cells[:, rep_time > 125]), rep_time[rep_time > 125], color="m", ax=axes[1][1])
    axes[1][1].set_xlabel("Time [s]")
    axes[1][1].set_ylabel("dF/F")
    sns.despine(fig)
    fig.tight_layout()
