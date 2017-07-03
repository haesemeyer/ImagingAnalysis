# Script to aggregate plots for Figure 1 of Heat-Imaging paper
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib as mpl
import h5py
import pickle
from mh_2P import MotorContainer, SLHRepeatExperiment, raster_plot, OpenStack, cutOutliers
from typing import List
from motorPredicates import bias, high_bias_bouts, unbiased_bouts
from multiExpAnalysis import dff
from scipy.stats import mode


def trial_average(m: np.ndarray, sum_it=False, n_trials=3):
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
    save_folder = "./HeatImaging/Figure1/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    membership = np.array(dfile['membership'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    mship_nonan = membership[no_nan_aa]
    all_activity = np.array(dfile['all_activity'])
    pstream = np.array(dfile['exp_data_pickle'])
    exp_data = pickle.loads(pstream)  # type: List[SLHRepeatExperiment]
    del pstream
    # limit sourceFiles and vertices to the contents of all_activity
    sourceFiles = [(g[0], e.original_time_per_frame) for e in exp_data for g in e.graph_info]
    verts = [g[1] for e in exp_data for g in e.graph_info]
    sourceFiles = [sf for i, sf in enumerate(sourceFiles) if no_nan_aa[i]]
    verts = [vs for i, vs in enumerate(verts) if no_nan_aa[i]]
    dfile.close()
    # create motor containers
    tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
    itime = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    mc_all_raw = MotorContainer(sourceFiles, itime, 0, hdf5_store=tailstore)
    mc_swims = MotorContainer(sourceFiles, itime, 0, predicate=unbiased_bouts, tdd=mc_all_raw.tdd)
    mc_flicks = MotorContainer(sourceFiles, itime, 0, predicate=high_bias_bouts, tdd=mc_all_raw.tdd)
    # initialize traces of mc_all_raw by obtaining average obtain other averages for plotting
    avg_all = mc_all_raw.avg_motor_output
    avg_swims = mc_swims.avg_motor_output
    avg_flicks = mc_flicks.avg_motor_output

    # plot motor raster plot
    motor_raster = np.vstack([mc_all_raw.traces[k][None, :] for k in mc_all_raw.traces.keys()])
    motor_raster = trial_average(motor_raster, True)
    fig, ax = pl.subplots()
    trial_time = np.arange(motor_raster.shape[1]) / 5
    raster_plot(motor_raster, trial_time, ax, 1)
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.set_xlim(0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Imaging plane")
    sns.despine(fig, ax)
    fig.savefig(save_folder+"motor_raster.pdf", type="pdf")

    # plot per-trial power at sample and measured temperature
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    p_at_samp = np.array(stim_file["sine_L_H_pwrAtSample"])
    p_at_samp = trial_average(np.add.reduceat(p_at_samp, np.arange(0, p_at_samp.size, 20 // 5))).ravel() / (20 // 5)
    t_at_samp = np.array(stim_file["sine_L_H_temp"])
    t_at_samp = trial_average(np.add.reduceat(t_at_samp, np.arange(0, t_at_samp.size, 20 // 5))).ravel() / (20 // 5)
    fig, ax = pl.subplots()
    ax.plot(trial_time, p_at_samp, 'k', lw=0.75)
    ax.set_ylabel("Power at sample [mW]")
    ax.set_xlabel("Time [s]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax2 = ax.twinx()
    ax2.plot(trial_time, t_at_samp, 'r')
    ax2.set_ylabel("Temperature [C]", color='r')
    ax2.tick_params('y', colors='r')
    sns.despine(right=False)
    fig.savefig(save_folder+"stimulus.pdf", type="pdf")
    stim_file.close()

    # plot all-activity heatmap sorted into ON and OFF
    cells_on = all_activity[np.logical_and(mship_nonan > -1, mship_nonan < 4), :]
    t_avg_on = trial_average(cells_on)
    cells_off = all_activity[np.logical_and(mship_nonan > 3, mship_nonan < 6), :]
    t_avg_off = trial_average(cells_off)
    act_to_plot = np.vstack((dff(t_avg_on), dff(t_avg_off)))
    fig, ax = pl.subplots()
    sns.heatmap(act_to_plot, xticklabels=150, yticklabels=5000, vmin=-3, vmax=3, ax=ax, rasterized=True)
    fig.savefig(save_folder+"activity_heatmap.pdf", type="pdf")

    # plot activity of example neuron
    ex_index = 100
    fig, ax = pl.subplots()
    ax.plot(trial_time, dff(t_avg_on)[ex_index, :])
    ax.set_xlabel("Time [s]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.set_ylabel("dF/F0")
    sns.despine(fig, ax)
    fig.savefig(save_folder+"neuron_example_trace.pdf", type="pdf")
    # also plot this neurons imaging plane projection as well as the ROI
    on = np.logical_and(mship_nonan > -1, mship_nonan < 4)
    sfile = [sf for i, sf in enumerate(sourceFiles) if on[i]][ex_index][0]
    vertices = [vs for i, vs in enumerate(verts) if on[i]][ex_index]
    tstack = OpenStack(sfile)
    tstack = np.sum(tstack, 0)
    tstack = cutOutliers(tstack, 99.9)
    projection = np.zeros((tstack.shape[0], tstack.shape[1], 3))
    projection[:, :, 0] = tstack * 0.8
    projection[:, :, 1] = tstack * 0.8
    projection[:, :, 2] = tstack * 0.8
    for v in vertices:
        projection[v[0], v[1], 1] = 0.651
        projection[v[0], v[1], 2] = 0.318
    fig, ax = pl.subplots()
    ax.imshow(projection)
    fig.savefig(save_folder+"example_neuron_location.pdf", type="pdf", dpi=300)

    # plot example trail-trace
    mot_trial_time = np.linspace(0, trial_time.max(), int(trial_time.max() * 100), endpoint=False)
    fig, ax = pl.subplots()
    ax.plot(mot_trial_time, mc_all_raw.tdd[sourceFiles[ex_index]].cumAngles[:mot_trial_time.size])
    ax.set_xlabel("Time [s]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.set_ylabel("Tail angle [deg]")
    ax.set_ylim(-150, 150)
    sns.despine(fig, ax)
    fig.savefig(save_folder+"motor_example_trace.pdf", type="pdf")

    # plot bias histogram of all bouts
    biases = []
    for k in mc_all_raw.tdd.fileNames:
        td = mc_all_raw.tdd[k]
        mca = mode(td.cumAngles)[0]
        if td.bouts is not None:
            for b in td.bouts.astype(int):
                biases.append(bias(b[0], b[1]+1, td.cumAngles, mca))
    biases = np.array(biases)
    biases = biases[np.logical_not(np.isnan(biases))]
    fig, ax = pl.subplots()
    counts, bedges, patches = ax.hist(biases, 100)
    bcents = bedges[:-1] + np.diff(bedges)/2
    for b, p in zip(bcents, patches):
        if np.abs(b) < 0.8:
            p.set_facecolor("C0")
        else:
            p.set_facecolor("C1")
    ax.set_yscale('Log')
    ax.plot([-0.8, -0.8], [1e2, counts.max()], 'k--')
    ax.plot([0.8, 0.8], [1e2, counts.max()], 'k--')
    ax.set_ylim(1e2)
    sns.despine(fig, ax)
    fig.savefig(save_folder+"bout_bias_histogram.pdf", type="pdf")

    # plot average frequency of flicks and bouts
    t_avg_swims = trial_average(avg_swims).ravel() * 5
    t_avg_flicks = trial_average(avg_flicks).ravel() * 5
    fig, ax = pl.subplots()
    ax.plot(trial_time, t_avg_swims, label="Swim bouts")
    ax.plot(trial_time, t_avg_flicks, label="Strong flicks")
    ax.set_xlim(0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bout frequency [Hz]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder+"motor_type_out.pdf", type="pdf")
