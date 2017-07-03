# Script to aggregate plots for Figure 7 of Heat-Imaging paper
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
import matplotlib as mpl
from Figure3 import trial_average, dff
from analyzeSensMotor import RegionResults
from typing import Dict
from sensMotorModel import ModelResult, standardize, cubic_nonlin, run_model
from multiExpAnalysis import max_cluster


def col_std(m):
    avg = np.mean(m, 0, keepdims=True)
    s = np.std(m, 0, keepdims=True)
    return (m - avg) / s


if __name__ == "__main__":
    save_folder = "./HeatImaging/Figure7/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load region results
    test_labels = ["Trigeminal", "Rh6"]
    region_results = {}  # type: Dict[str, RegionResults]
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for k in test_labels:
        region_results[k] = (pickle.loads(np.array(storage[k])))
    storage.close()
    # load model results
    model_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/model_170702.hdf5', 'r')
    model_results = pickle.loads(np.array(model_file["model_results"]))  # type: Dict[str, ModelResult]
    stim_in = np.array(model_file["stim_in"])
    m_in = np.array(model_file["m_in"])
    s_in = np.array(model_file["s_in"])
    model_file.close()
    names_tg = ["TG_ON", "TG_OFF"]
    names_rh6 = ["Fast_ON", "Slow_ON", "Fast_OFF", "Slow_OFF", "Delayed_OFF"]
    motor_res_names = ["M_All", "M_Flick", "M_Swim", "M_StimOn", "M_NoStim"]
    # load motor results
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/motor_system.hdf5", "r")
    motor_type_regs = standardize(trial_average(np.array(motor_store["motor_type_regs"]).T, 3)).T
    flick_out = standardize(trial_average(np.array(motor_store["flick_out"]), 3)).ravel()
    swim_out = standardize(trial_average(np.array(motor_store["swim_out"]), 3)).ravel()
    motor_store.close()
    # load detail char stimulus
    t_time_sinLH = np.arange(region_results["Trigeminal"].regressors.shape[0]) / 5.0
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    dt_t_at_samp = np.array(stim_file["detail_char_temp"])
    dt_t_at_samp = trial_average(np.add.reduceat(dt_t_at_samp,
                                                 np.arange(0, dt_t_at_samp.size, 20 // 5)), 10).ravel() / (20 // 5)
    dt_t_at_samp = (dt_t_at_samp-m_in) / s_in
    stim_file.close()
    t_time_dtChar = np.arange(dt_t_at_samp.size) / 5.0

    # plot prediction of swims and flicks of the original sine lo hi data
    sw_sinLH, fl_sinLH = run_model(stim_in, model_results)[:2]
    fig, (ax_sw, ax_flk) = pl.subplots(ncols=2, sharex=True, sharey=True)
    ax_sw.plot(t_time_sinLH, swim_out, 'k', label="Swims")
    ax_sw.plot(t_time_sinLH, standardize(sw_sinLH), "C0", label="Swim prediction")
    ax_sw.set_xlabel("Time [s]")
    ax_sw.set_ylabel("Motor output [AU]")
    ax_sw.set_title("R2 = {0:.2}".format(np.corrcoef(sw_sinLH, swim_out)[0, 1] ** 2))
    ax_sw.legend()
    ax_flk.plot(t_time_sinLH, flick_out, 'k', label="Flicks")
    ax_flk.plot(t_time_sinLH, standardize(fl_sinLH), "C1", label="Flick prediction")
    ax_flk.set_xlabel("Time [s]")
    ax_flk.set_title("R2 = {0:.2}".format(np.corrcoef(fl_sinLH, flick_out)[0, 1] ** 2))
    ax_flk.legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder+"SineLH_Motor_Prediction.pdf", type="pdf")

    # plot prediction of swims and flicks for detail char experiments
    detChar_swims = np.load("detailChar_swims.npy")
    detChar_flicks = np.load("detailChar_flicks.npy")
    sw_dtCh, fl_dtCh, rh6_dtCh = run_model(dt_t_at_samp, model_results)
    no_tap_inf = np.logical_and(t_time_dtChar > 10, t_time_dtChar < 128)
    fig, (ax_sw, ax_flk) = pl.subplots(ncols=2, sharex=True, sharey=True)
    ax_sw.plot(t_time_dtChar[no_tap_inf], standardize(detChar_swims[no_tap_inf]), 'k', label="Swims")
    ax_sw.plot(t_time_dtChar[no_tap_inf], standardize(sw_dtCh[no_tap_inf]), "C0", label="Swim prediction")
    ax_sw.set_xlabel("Time [s]")
    ax_sw.set_ylabel("Motor output [AU]")
    ax_sw.set_title("R2 = {0:.2}".format(np.corrcoef(detChar_swims[no_tap_inf], sw_dtCh[no_tap_inf])[0, 1] ** 2))
    ax_sw.legend()
    ax_flk.plot(t_time_dtChar[no_tap_inf], standardize(detChar_flicks[no_tap_inf]), 'k', label="Flicks")
    ax_flk.plot(t_time_dtChar[no_tap_inf], standardize(fl_dtCh[no_tap_inf]), "C1", label="Flick prediction")
    ax_flk.set_xlabel("Time [s]")
    ax_flk.set_title("R2 = {0:.2}".format(np.corrcoef(detChar_flicks[no_tap_inf], fl_dtCh[no_tap_inf])[0, 1] ** 2))
    ax_flk.legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "DetailChar_Motor_Prediction.pdf", type="pdf")

    # use predicted rh6 activity in detail char experiments to cluster Rh6 data from those experiments into our types
    dfile = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/detailChar_data.hdf5", "r")
    dt_act = np.array(dfile["all_activity"])
    dt_regions = pickle.loads(np.array(dfile["all_rl_pickle"]))
    dfile.close()
    rh6_act = dt_act[(dt_regions == "Rh_6").ravel(), :]
    # create trial averages of rh6 activity
    ta_rh6_act = np.mean(rh6_act.reshape((rh6_act.shape[0], 25, rh6_act.shape[1] // 25)), 1)
    # create correlation matrix for correlations of real activity to predicted rh6 activity as regressors
    pred_reg_corr_mat = np.zeros((ta_rh6_act.shape[0], rh6_dtCh.shape[1]))
    for i in range(rh6_dtCh.shape[1]):
        reg = rh6_dtCh[:, i]
        for j, a in enumerate(ta_rh6_act):
            pred_reg_corr_mat[j, i] = np.corrcoef(a, reg)[0, 1]
    sig_cells = (np.sum(pred_reg_corr_mat >= 0.6, 1) > 0)
    dt_sig_corrs = pred_reg_corr_mat[sig_cells, :]
    activity_sig_corrs = dff(ta_rh6_act[sig_cells, :])
    mclust = max_cluster(np.argmax(dt_sig_corrs, 1))
    # plot clusters
    fig, ax = pl.subplots()
    sns.heatmap(dt_sig_corrs[np.argsort(mclust.labels_), :], yticklabels=50,
                xticklabels=["Fast ON", "Slow ON", "Fast OFF", "Slow OFF", "Dld. OFF"], ax=ax)
    # plot cluster boundaries
    covered = 0
    for i in range(pred_reg_corr_mat.shape[1]):
        covered += np.sum(mclust.labels_ == i)
        ax.plot([0, dt_sig_corrs.shape[1] + 1], [mclust.labels_.size - covered, mclust.labels_.size - covered], 'k')
    ax.set_ylabel("Cells")
    fig.savefig(save_folder + "DetailChar_Rh6_clustering.pdf", type="pdf")

    # plot average ON cluster activity
    fig, ax = pl.subplots()
    sns.tsplot(activity_sig_corrs[mclust.labels_ == 0, :], t_time_dtChar, color="C0")
    sns.tsplot(activity_sig_corrs[mclust.labels_ == 1, :], t_time_dtChar, color="C1")
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("dF/F")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "DetailChar_Rh6_ON_activity.pdf", type="pdf")

    # plot average OFF cluster activity
    fig, ax = pl.subplots()
    sns.tsplot(activity_sig_corrs[mclust.labels_ == 2, :], t_time_dtChar, color="C0")
    sns.tsplot(activity_sig_corrs[mclust.labels_ == 3, :], t_time_dtChar, color="C1")
    sns.tsplot(activity_sig_corrs[mclust.labels_ == 4, :], t_time_dtChar, color="C2")
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("dF/F")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "DetailChar_Rh6_OFF_activity.pdf", type="pdf")

    # plot each predicted regressor together with the cluster average
    # Fast ON
    fig, ax = pl.subplots()
    ax.plot(t_time_dtChar, standardize(rh6_dtCh[:, 0]), 'k')
    ax.plot(t_time_dtChar, standardize(np.mean(activity_sig_corrs[mclust.labels_ == 0, :], 0)), "C3")
    ax.plot([129.9, 129.9], [-1, 3], 'k--')
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activity [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "DetChar_FastON_vs_reg.pdf", type="pdf")
    # Slow ON
    fig, ax = pl.subplots()
    ax.plot(t_time_dtChar, standardize(rh6_dtCh[:, 1]), 'k')
    ax.plot(t_time_dtChar, standardize(np.mean(activity_sig_corrs[mclust.labels_ == 1, :], 0)), "C1")
    ax.plot([129.9, 129.9], [-1, 3], 'k--')
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activity [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "DetChar_SlowON_vs_reg.pdf", type="pdf")
    # Fast OFF
    fig, ax = pl.subplots()
    ax.plot(t_time_dtChar, standardize(rh6_dtCh[:, 2]), 'k')
    ax.plot(t_time_dtChar, standardize(np.mean(activity_sig_corrs[mclust.labels_ == 2, :], 0)), "C2")
    ax.plot([129.9, 129.9], [-1, 3], 'k--')
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activity [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "DetChar_FastOFF_vs_reg.pdf", type="pdf")
    # Slow OFF
    fig, ax = pl.subplots()
    ax.plot(t_time_dtChar, standardize(rh6_dtCh[:, 3]), 'k')
    ax.plot(t_time_dtChar, standardize(np.mean(activity_sig_corrs[mclust.labels_ == 3, :], 0)), "C0")
    ax.plot([129.9, 129.9], [-1, 3], 'k--')
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activity [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "DetChar_SlowOFF_vs_reg.pdf", type="pdf")
    # Delayed OFF
    fig, ax = pl.subplots()
    ax.plot(t_time_dtChar, standardize(rh6_dtCh[:, 4]), 'k')
    ax.plot(t_time_dtChar, standardize(np.mean(activity_sig_corrs[mclust.labels_ == 4, :], 0)), "C5")
    ax.plot([129.9, 129.9], [-1, 3], 'k--')
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activity [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "DetChar_DldOFF_vs_reg.pdf", type="pdf")
