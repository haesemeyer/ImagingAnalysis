# Script to aggregate plots for Supplemental Figure 7 of Heat-Imaging paper
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
import matplotlib as mpl
from Figure3 import trial_average
from analyzeSensMotor import RegionResults
from typing import Dict
from sensMotorModel import ModelResult, standardize, run_model


def col_std(m):
    avg = np.mean(m, 0, keepdims=True)
    s = np.std(m, 0, keepdims=True)
    return (m - avg) / s


if __name__ == "__main__":
    save_folder = "./HeatImaging/FigureS7/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load region results
    test_labels = ["Trigeminal", "Rh6"]
    region_results = {}  # type: Dict[str, RegionResults]
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for k in test_labels:
        region_results[k] = (pickle.loads(np.array(storage[k])))
    storage.close()
    # load rh6-no-filter model results
    model_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/model_noRh6Filts.hdf5', 'r')
    model_results = pickle.loads(np.array(model_file["model_results"]))  # type: Dict[str, ModelResult]
    stim_in = np.array(model_file["stim_in"])
    m_in = np.array(model_file["m_in"])
    s_in = np.array(model_file["s_in"])
    model_file.close()
    names_tg = ["TG_ON", "TG_OFF"]
    names_rh6 = ["Slow_ON", "Slow_OFF", "Fast_ON",  "Fast_OFF", "Delayed_OFF"]
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

    # plot prediction of swims and flicks in the absence of any temporal filtering in Rh6

    # sine-lo-hi
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
    fig.savefig(save_folder + "SineLH_noRh6filt_motPred.pdf", type="pdf")

    # detail char experiments
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
    fig.savefig(save_folder + "DetailChar_noRh6filt_motPred.pdf", type="pdf")

    # use fourier transform to compare stimulus dynamics raw and after filtering by our trigeminal filter
    dt_t_at_samp -= np.mean(dt_t_at_samp)
    stim_in -= np.mean(stim_in)
    tg_filter = model_results["TG_ON"].filter_coefs
    fft_dt_raw = np.fft.rfft(dt_t_at_samp)
    conv = np.convolve(dt_t_at_samp, tg_filter)[:dt_t_at_samp.size]
    conv -= np.mean(conv)
    fft_dt_filt = np.fft.rfft(conv)
    freq_dt = np.fft.rfftfreq(dt_t_at_samp.size, 1./5)
    fft_slh_raw = np.fft.rfft(stim_in)
    conv = np.convolve(stim_in, tg_filter)[:stim_in.size]
    conv -= np.mean(conv)
    fft_slh_filt = np.fft.rfft(conv)
    freq_slh = np.fft.rfftfreq(stim_in.size, 1./5)
    abs_dt_raw = np.absolute(fft_dt_raw) / fft_dt_raw.size
    abs_dt_filt = np.absolute(fft_dt_filt) / fft_dt_filt.size
    abs_slh_raw = np.absolute(fft_slh_raw) / fft_slh_raw.size
    abs_slh_filt = np.absolute(fft_slh_filt) / fft_slh_filt.size

    fig, (ax_r, ax_f) = pl.subplots(ncols=2)
    ax_r.plot(freq_slh, np.cumsum(abs_slh_raw)/abs_slh_raw.sum(), label="SLH")
    ax_r.plot(freq_dt, np.cumsum(abs_dt_raw) / abs_dt_raw.sum(), label="dtChar")
    ax_r.legend()
    ax_r.set_xlabel("Frequency")
    ax_r.set_ylabel("Cumulative proportion")
    ax_f.plot(freq_slh, np.cumsum(abs_slh_filt)/abs_slh_filt.sum(), label="SLH")
    ax_f.plot(freq_dt, np.cumsum(abs_dt_filt) / abs_dt_filt.sum(), label="dtChar")
    ax_f.set_xlabel("Frequency")
    ax_f.set_ylabel("Cumulative proportion")
    ax_f.legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "Stimulus_spectrum_compare.pdf", type="pdf")
