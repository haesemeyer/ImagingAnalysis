import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib as mpl
import pandas
import h5py
import pickle
from multiDetailCharAnalysis import cut_neg, cut_pos


def dff(ts):
    # f0 = np.percentile(ts, 10, axis=1, keepdims=True)
    f0 = np.mean(ts[:, 10*5:20*5], axis=1, keepdims=True)
    f0[f0 < 0.05] = 0.05
    return (ts-f0)/f0


def trial_average(m: np.ndarray, n_trials=25, sum_it=False):
    if m.ndim == 2:
        m_t = np.reshape(m, (m.shape[0], n_trials, m.shape[1]//n_trials))
    elif m.ndim == 1:
        m_t = np.reshape(m, (1, n_trials, m.shape[0] // n_trials))
    else:
        raise ValueError("m has to be either 1 or 2-dimensional")
    if sum_it:
        return np.sum(m_t, 1)
    return np.mean(m_t, 1)


if __name__ == "__main__":
    save_folder = "./HeatImaging/Figure3/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load main data
    dfile = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/detailChar_data.hdf5", "r")
    all_rl = pickle.loads(np.array(dfile["all_rl_pickle"]))
    on_no_tap_cells = np.array(dfile["on_no_tap_cells"])
    on_tap_cells = np.array(dfile["on_tap_cells"])
    off_no_tap_cells = np.array(dfile["off_no_tap_cells"])
    tap_only = np.array(dfile["tap_only"])
    stim_units = np.array(dfile["stim_units"])
    act_sign = np.array(dfile["act_sign"])
    p_tap = np.array(dfile["p_tap"])
    rep_time = np.array(dfile["rep_time"])
    avg_motor = np.array(dfile["motor_all_raw"])
    avg_motor = trial_average(avg_motor).ravel() * 5
    tu_avg = np.array(dfile["tu_avg"])
    dfile.close()
    # load laser and heat stimulus
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    p_at_samp = np.array(stim_file["detalChar_pwrAtSample"])
    p_at_samp = trial_average(np.add.reduceat(p_at_samp, np.arange(0, p_at_samp.size, 20 // 5)), 10).ravel() / (20 // 5)
    t_at_samp = np.array(stim_file["detail_char_temp"])
    t_at_samp = trial_average(np.add.reduceat(t_at_samp, np.arange(0, t_at_samp.size, 20 // 5)), 10).ravel() / (20 // 5)
    stim_file.close()

    # plot stimulus
    fig, ax = pl.subplots()
    ax.plot(rep_time, p_at_samp, 'k', lw=0.75)
    ax.set_ylabel("Power at sample [mW]")
    ax.set_xlabel("Time [s]")
    ax.set_xticks([0, 30, 60, 90, 120])
    ax2 = ax.twinx()
    ax2.plot(rep_time, t_at_samp, 'r')
    ax2.set_ylabel("Temperature [C]", color='r')
    ax2.tick_params('y', colors='r')
    sns.despine(right=False)
    fig.savefig(save_folder+"stimulus.pdf", type="pdf")

    # plot stimulus inset split into heat and tap times
    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    ax_heat.plot(rep_time[rep_time < 125], t_at_samp[rep_time < 125])
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("Temperature [C]")
    ax_heat.set_xticks([0, 30, 60, 90, 120])
    ax_heat.set_ylim(23, 29)
    ax_tap.plot(rep_time[rep_time >= 125], t_at_samp[rep_time >= 125])
    ax_tap.set_xlabel("Time [s]")
    ax_tap.set_ylabel("Temperature [C]")
    ax_tap.set_xticks([125, 130, 135])
    ax_tap.plot([129.9, 129.9], [23, 29], "k--")
    ax_tap.set_ylim(23, 29)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "stimulus_inset.pdf", type="pdf")

    # plot overall motor output
    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]})
    ax_heat.plot(rep_time[rep_time < 125], avg_motor[rep_time < 125])
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("Bout frequency [Hz]")
    ax_heat.set_xticks([0, 30, 60, 90, 120])
    ax_tap.plot(rep_time[rep_time >= 125], avg_motor[rep_time >= 125])
    ax_tap.set_xlabel("Time [s]")
    ax_tap.set_ylabel("Bout frequency [Hz]")
    ax_tap.set_xticks([125, 130, 135])
    # ax_tap.plot([129.8, 129.8], [0, 5], "k--")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "avg_motor_output.pdf", type="pdf")

    # plot average activity of the types segregated by ON/OFF with heat and tap responses in separate plots
    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(on_tap_cells)[:, rep_time < 125], rep_time[rep_time < 125], color="r", ax=ax_heat, ci=68)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    ax_heat.set_xticks([0, 30, 60, 90, 120])
    sns.tsplot(dff(on_tap_cells)[:, rep_time >= 125], rep_time[rep_time >= 125], color="r", ax=ax_tap, ci=68)
    ax_tap.set_xlabel("Time [s]")
    ax_tap.set_xticks([125, 130, 135])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "on_tap_cells.pdf", type="pdf")

    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(on_no_tap_cells)[:, rep_time < 125], rep_time[rep_time < 125], color="orange", ax=ax_heat, ci=68)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    ax_heat.set_xticks([0, 30, 60, 90, 120])
    sns.tsplot(dff(on_no_tap_cells)[:, rep_time >= 125], rep_time[rep_time >= 125], color="orange", ax=ax_tap, ci=68)
    ax_tap.set_xlabel("Time [s]")
    ax_tap.set_xticks([125, 130, 135])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "on_no_tap_cells.pdf", type="pdf")

    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(off_no_tap_cells)[:, rep_time < 125], rep_time[rep_time < 125], color="b", ax=ax_heat, ci=68)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    ax_heat.set_xticks([0, 30, 60, 90, 120])
    sns.tsplot(dff(off_no_tap_cells)[:, rep_time >= 125], rep_time[rep_time >= 125], color="b", ax=ax_tap, ci=68)
    ax_tap.set_xlabel("Time [s]")
    ax_tap.set_xticks([125, 130, 135])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "off_no_tap_cells.pdf", type="pdf")

    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    sns.tsplot(dff(tu_avg)[:, rep_time < 125], rep_time[rep_time < 125], color="k", ax=ax_heat, ci=68)
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("dF/F0")
    ax_heat.set_xticks([0, 30, 60, 90, 120])
    sns.tsplot(dff(tu_avg)[:, rep_time >= 125], rep_time[rep_time >= 125], color="k", ax=ax_tap, ci=68)
    ax_tap.set_xlabel("Time [s]")
    ax_tap.set_xticks([125, 130, 135])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "tap_cells.pdf", type="pdf")

    # plot stimulus representation preference
    plot_order = ["TG", "Pallium", "Hab", "Cerebellum", "Rh_6"]
    heat_pi = {}
    tap_pi = {}
    pure_heat = all_rl[stim_units][np.logical_and(act_sign[stim_units] != 0, p_tap[stim_units] < cut_neg)]
    pure_tap = all_rl[tap_only]
    mixed = all_rl[stim_units][np.logical_and(act_sign[stim_units] != 0, p_tap[stim_units] > cut_pos)]
    for i, r in enumerate(np.unique(all_rl)):
        if r == "" or r == "vMB" or r == "HB":
            continue
        phr = np.sum(pure_heat == r)
        ptr = np.sum(pure_tap == r)
        mr = np.sum(mixed == r)
        heat_pi[r] = [phr / (phr + mr)]
        tap_pi[r] = [ptr / (ptr + mr)]

    fig, ax = pl.subplots()
    sns.barplot(data=pandas.DataFrame(tap_pi), order=plot_order, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction tap only")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "pi_tap.pdf", type="pdf")

    fig, ax = pl.subplots()
    sns.barplot(data=pandas.DataFrame(heat_pi), order=plot_order, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction heat only")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "pi_heat.pdf", type="pdf")
