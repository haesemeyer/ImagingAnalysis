import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from mh_2P import DetailCharExperiment, MotorContainer, UiGetFile, assign_region_label, RegionContainer
from motorPredicates import high_bias_bouts, unbiased_bouts, left_bias_bouts, right_bias_bouts
import pickle
from typing import List
from scipy.stats import wilcoxon
import h5py
from os import path
import matplotlib as mpl
import pandas


def dff(ts):
    # f0 = np.percentile(ts, 10, axis=1, keepdims=True)
    f0 = np.mean(ts[:, 10*5:20*5], axis=1, keepdims=True)
    f0[f0 < 0.05] = 0.05
    return (ts-f0)/f0


if __name__ == "__main__":
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # Load data files
    print("Load data files", flush=True)
    exp_data = []  # type: List[DetailCharExperiment]
    dfnames = UiGetFile([('Experiment data', '.pickle')], True)
    for name in dfnames:
        f = open(name, 'rb')
        d = pickle.load(f)
        exp_data.append(d)
        f.close()
    # generate our (sourceFile, ca-frame-time) tuple list for the motor containers
    source_files = []
    var_scores = np.array([])
    all_activity = np.array([])
    all_rl = []  # for each cell the labeled region it is coming from
    region_dict = {}
    for i, data in enumerate(exp_data):
        source_files += [(gi[0], data.original_time_per_frame) for gi in data.graph_info]
        var_scores = np.r_[var_scores, data.variance_score()]
        if i == 0:
            # NOTE: RawData field is 64 bit float - convert to 32 when storing in all_activity
            all_activity = data.RawData.astype(np.float32)
        else:
            all_activity = np.r_[all_activity, data.RawData.astype(np.float32)]
        for gi in data.graph_info:
            if gi[0] not in region_dict:
                # first try to find the segmentation file in the same location as the source file
                ext_start = gi[0].find('.tif')
                segment_name = gi[0][:ext_start] + "_SEG.hdf5"
                if not path.exists(segment_name):
                    # look in same directory as experiment data files
                    name = path.split(gi[0])[1]
                    directory = path.split(dfnames[0])[0]
                    ext_start = name.find('.tif')
                    name = name[:ext_start] + "_SEG.hdf5"
                    segment_name = path.join(directory, name)
                cont_file = h5py.File(segment_name, 'r')
                region_dict[gi[0]] = RegionContainer.load_container_list(cont_file)
                cont_file.close()
            region_list = region_dict[gi[0]]
            # not the first vertex entry is the y-coordinate (=row of matrix),
            # the second the x-coordinate (=column of matrix)
            centroid = np.array([np.mean([v[1] for v in gi[1]]), np.mean([v[0] for v in gi[1]]), 0])
            all_rl.append(assign_region_label([centroid], region_list, 1, 1))
    all_rl = np.array(all_rl)
    var_scores[np.isnan(var_scores)] = 0
    # create motor containers
    i_time = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    tc = exp_data[0].caTimeConstant
    mc_all = MotorContainer(source_files, i_time, tc)
    mc_high_bias = MotorContainer(source_files, i_time, tc, high_bias_bouts, tdd=mc_all.tdd)
    mc_low_bias = MotorContainer(source_files, i_time, tc, unbiased_bouts, tdd=mc_all.tdd)
    mc_left_bias = MotorContainer(source_files, i_time, tc, left_bias_bouts, tdd=mc_all.tdd)
    mc_right_bias = MotorContainer(source_files, i_time, tc, right_bias_bouts, tdd=mc_all.tdd)
    mc_hb_raw = MotorContainer(source_files, i_time, 0, high_bias_bouts, tdd=mc_all.tdd)
    mc_lb_raw = MotorContainer(source_files, i_time, 0, unbiased_bouts, tdd=mc_all.tdd)
    mc = [mc_all, mc_high_bias, mc_low_bias, mc_left_bias, mc_right_bias]
    mc_all_raw = MotorContainer(source_files, i_time, 0, tdd=mc_all.tdd)

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
    is_pot_stim = np.logical_and(var_scores >= 0.1, np.max(mc_type_corrs, 1) < 0.4)
    # only consider cells in marked regions as potential stimulus units
    is_pot_stim = np.logical_and(is_pot_stim, all_rl.ravel() != "")

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
    p_stp[np.isnan(p_stp)] = 1
    p_sin[np.isnan(p_sin)] = 1
    p_tap[np.isnan(p_tap)] = 1
    # mark units that show significant change in both periods
    sig_act = np.logical_and(p_stp < 0.05, p_sin < 0.05)
    stim_units = np.logical_and(sig_act, is_pot_stim)
    # compute repeat-average of all stim_units
    su_avg = np.mean(data.repeat_align(all_activity[stim_units, :]), 1)

    # mark units that only respond to taps and compute their repeat average
    tap_act = np.logical_and(p_tap < 0.01, is_pot_stim)
    not_heat = np.logical_and(p_stp > 0.05, p_sin > 0.05)
    tap_only = np.logical_and(tap_act, not_heat)
    tu_avg = np.mean(data.repeat_align(all_activity[tap_only, :]), 1)

    # divide cells based on heat sign and whether they also significantly respond to taps or not
    on_tap_cells = su_avg[np.logical_and(act_sign[stim_units] > 0, p_tap[stim_units] < 0.05), :]
    on_no_tap_cells = su_avg[np.logical_and(act_sign[stim_units] > 0, p_tap[stim_units] > 0.05), :]
    off_tap_cells = su_avg[np.logical_and(act_sign[stim_units] < 0, p_tap[stim_units] < 0.05), :]
    off_no_tap_cells = su_avg[np.logical_and(act_sign[stim_units] < 0, p_tap[stim_units] > 0.05), :]

    # plot average activity of the types segregated by ON/OFF with heat and tap responses in separate plots
    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(on_tap_cells)[:, rep_time < 125], rep_time[rep_time < 125], color="r", ax=ax_heat, ci=95)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    sns.tsplot(dff(on_tap_cells)[:, rep_time >= 125], rep_time[rep_time >= 125], color="r", ax=ax_tap, ci=95)
    ax_tap.set_xlabel("Time [s]")
    sns.despine(fig)
    fig.tight_layout()

    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(on_no_tap_cells)[:, rep_time < 125], rep_time[rep_time < 125], color="orange", ax=ax_heat, ci=95)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    sns.tsplot(dff(on_no_tap_cells)[:, rep_time >= 125], rep_time[rep_time >= 125], color="orange", ax=ax_tap, ci=95)
    ax_tap.set_xlabel("Time [s]")
    sns.despine(fig)
    fig.tight_layout()

    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(off_no_tap_cells)[:, rep_time < 125], rep_time[rep_time < 125], color="b", ax=ax_heat, ci=95)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    sns.tsplot(dff(off_no_tap_cells)[:, rep_time >= 125], rep_time[rep_time >= 125], color="b", ax=ax_tap, ci=95)
    ax_tap.set_xlabel("Time [s]")
    sns.despine(fig)
    fig.tight_layout()

    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(tu_avg)[:, rep_time < 125], rep_time[rep_time < 125], color="k", ax=ax_heat, ci=95)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    sns.tsplot(dff(tu_avg)[:, rep_time >= 125], rep_time[rep_time >= 125], color="k", ax=ax_tap, ci=95)
    ax_tap.set_xlabel("Time [s]")
    sns.despine(fig)
    fig.tight_layout()

    # simplify our annotation
    all_rl[all_rl == "Cerebellum_L"] = "Cerebellum"
    all_rl[all_rl == "Cerebellum_R"] = "Cerebellum"
    all_rl[all_rl == "HB_L"] = "HB"
    all_rl[all_rl == "HB_R"] = "HB"
    all_rl[all_rl == "RH6_R"] = "Rh_6"
    all_rl[all_rl == "RH6_L"] = "Rh_6"
    all_rl[all_rl == "Rh6_L"] = "Rh_6"
    all_rl[all_rl == "Rh6_R"] = "Rh_6"

    # plot counts of different types across regions
    fig, ax = pl.subplots()
    sns.countplot(sorted([str(a) for a in all_rl[tap_only]]), ax=ax)
    ax.set_title("Tap only")
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    sns.countplot(sorted([str(a) for a in all_rl[stim_units][np.logical_and(act_sign[stim_units] > 0,
                                                                     p_tap[stim_units] > 0.05)]]), ax=ax)
    ax.set_title("Heat only")
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    sns.countplot(sorted([str(a) for a in all_rl[stim_units][np.logical_and(act_sign[stim_units] > 0,
                                                                     p_tap[stim_units] < 0.05)]]), ax=ax)
    ax.set_title("Heat and tap")
    sns.despine(fig, ax)

    # for each region compute it's preference for representing heat and tap only versus mixed
    heat_pi = {}
    tap_pi = {}
    pure_heat = all_rl[stim_units][np.logical_and(act_sign[stim_units] > 0, p_tap[stim_units] > 0.05)]
    pure_tap = all_rl[tap_only]
    mixed = all_rl[stim_units][np.logical_and(act_sign[stim_units] > 0, p_tap[stim_units] < 0.05)]
    for i, r in enumerate(np.unique(all_rl)):
        if r == "":
            continue
        phr = np.sum(pure_heat == r)
        ptr = np.sum(pure_tap == r)
        mr = np.sum(mixed == r)
        heat_pi[r] = [(phr - mr) / (phr + mr)]
        tap_pi[r] = [(ptr - mr) / (ptr + mr)]

    pl.figure()
    sns.barplot(data=pandas.DataFrame(tap_pi))
    pl.ylim(-1, 1)
    pl.ylabel("Preference tap only")
    sns.despine()

    pl.figure()
    sns.barplot(data=pandas.DataFrame(heat_pi))
    pl.ylim(-1, 1)
    pl.ylabel("Preference heat only")
    sns.despine()
