from mh_2P import OpenStack, TailData, UiGetFile, NucGraph, CorrelationGraph, SOORepeatExperiment, SLHRepeatExperiment
from mh_2P import MakeNrrdHeader
import numpy as np
import nimfa
from scipy.signal import savgol_filter

import matplotlib.pyplot as pl
import seaborn as sns

import pickle
import nrrd

def dff(fluomat):
    f0 = np.median(fluomat[:, :144], 1, keepdims=True)
    f0[f0 == 0] = 0.1
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


def GetExperimentBaseName(exp_data):
    fullName = exp_data.graph_info[0][0]
    plane_start = fullName.find('_Z_')
    return fullName[:plane_start]


def MakeAndSaveROIStack(experiment_data, exp_zoom_factor, unit_cluster_ids, clusterNumber):
    """
    Creates ROI only (no anatomy background) stacks of all ROIs that belong
    to the given clusterNumber
    Args:
        experiment_data: The experiment for which to save the stack
        exp_zoom_factor: Zoom factor used during acquisition in order to save correct pixel sizes in nrrd file
        unit_cluster_ids: For each unit in experiment data its cluster id
        clusterNumber: The cluster number for which to create stack. If a list one stack per list will be saved
    """
    zstack = MakeMaskStack(experiment_data, (unit_cluster_ids == clusterNumber).astype(np.float32),
                           np.zeros(unit_cluster_ids.size), np.zeros(unit_cluster_ids.size), 0.5, 1.0)
    # recode and reformat zstack for nrrd saving
    nrrdStack = np.zeros((zstack.shape[2], zstack.shape[1], zstack.shape[0]), dtype=np.uint8, order='F')
    for i in range(zstack.shape[0]):
        nrrdStack[:, :, i] = (zstack[i, :, :, 0]*255).astype(np.uint8).T
    header = MakeNrrdHeader(nrrdStack, 500/512/exp_zoom_factor)
    assert np.isfortran(nrrdStack)
    out_name = GetExperimentBaseName(experiment_data) + '_C' + str(clusterNumber) + '.nrrd'
    nrrd.write(out_name, nrrdStack, header)


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
            all_motor = data.Vigor
        else:
            all_activity = np.r_[all_activity, data.RawData]
            all_motor = np.r_[all_motor, data.Vigor]
        p = data.computeFourierMetrics()[4]
        stim_phase = np.r_[stim_phase, p]

    # compute correlation matrix of all time-series data
    corr_mat = np.corrcoef(all_activity[is_pot_stim, :])
    # plot correlation heatmap sorted by number of correlated companions
    fig, ax = pl.subplots()
    sns.heatmap(corr_mat[:250, :250], xticklabels=50, yticklabels=50, ax=ax)
    pl.title('Cell to cell signal correlations examples')

    # generate plot of number of cells to analyze for different "other cell" and "n experiment" criteria
    n_exp_to_test = [0, 1, 2, 3, 4, 5]
    n_cells_to_test = list(range(30))
    exp_above = np.zeros((len(n_exp_to_test), corr_mat.shape[1]))
    cells_above = np.zeros((len(n_cells_to_test), corr_mat.shape[1]))
    for i, net in enumerate(n_exp_to_test):
        exp_above[i, :] = n_exp_r2_above_thresh(corr_mat, 0.5, exp_id[is_pot_stim]) >= net
    for i, nct in enumerate(n_cells_to_test):
        cells_above[i, :] = n_r2_above_thresh(corr_mat, 0.5) >= nct

    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        for i, net in enumerate(n_exp_to_test):
            n_remain = []
            for j, nct in enumerate(n_cells_to_test):
                n_remain.append(np.sum(np.logical_and(exp_above[i, :], cells_above[j, :])) / corr_mat.shape[0])
            ax.plot(n_cells_to_test, n_remain, 'o', label="At least "+str(net)+" other fish")
        ax.set_xlabel('Number of other cells with $R^2$ > 0.5')
        ax.set_ylabel('Fraction of stimulus cells to analyze')
        ax.set_xlim(-0.2)
        ax.set_yscale('log')
        ax.legend()

    # get all cells that have at least 20 other cells with a timeseries R2>0.5 and are spread
    # across at least 3 experiments
    exp_g_1 = n_exp_r2_above_thresh(corr_mat, 0.5, exp_id[is_pot_stim]) > 2
    c_g_9 = n_r2_above_thresh(corr_mat, 0.5) > 19
    to_analyze = np.logical_and(exp_g_1, c_g_9)
    analysis_data = all_activity[is_pot_stim, :][to_analyze, :]
    # marks, which of all units where used to derive regressors
    discovery_unit_marker = np.zeros(all_activity.shape[0], dtype=bool)
    discovery_unit_marker[is_pot_stim] = to_analyze

    del corr_mat

    # remove NaN containing traces from activity and motor matrix
    no_nan_aa = np.sum(np.isnan(all_activity), 1) == 0
    all_activity = all_activity[no_nan_aa, :]
    all_motor = all_motor[no_nan_aa, :]
    discovery_unit_marker = discovery_unit_marker[no_nan_aa]

    # analyze per-experiment swim vigors
    # all_expVigors = np.vstack([np.mean(expVigor(data), 0) for data in exp_data])
    # all_expVigors = all_expVigors / np.mean(all_expVigors, 1, keepdims=True)

    # TODO: The following is experiment specific - needs cleanup!
    avg_analysis_data = np.zeros((analysis_data.shape[0], analysis_data.shape[1] // 3))
    avg_analysis_data += analysis_data[:, :825]
    avg_analysis_data += analysis_data[:, 825:825*2]
    avg_analysis_data += analysis_data[:, 825*2:]
    norm_data = avg_analysis_data/np.percentile(avg_analysis_data, 20, 1, keepdims=True)  # F/F0
    # Note: currently we anyway normalize in NonNegMatFact so normalization above kinda useless
    n_regs = 10  # extract 5 components for our regressors
    nnmf, fit, fildata = NonNegMatFact(norm_data, 5, n_regs)
    W = np.array(nnmf.W)  # cells / components weight matrix
    H = np.array(nnmf.H)
    # plot connectivity matrix - which timepoints are encoded similarly
    fix, ax = pl.subplots()
    sns.heatmap(nnmf.connectivity(), xticklabels=50, yticklabels=50, cmap='bone_r', ax=ax)
    ax.set_title('Time connectivity')

    # extract our stimulus regressors by aggresively thresholding at the 95th percentile for weights
    # also plot the corresponding cluster means
    reg_orig = np.empty((analysis_data.shape[1], n_regs))
    time = np.arange(reg_orig.shape[0]) / 5
    for i in range(n_regs):
        cut = np.percentile(W[:, i], 95)
        reg_orig[:, i] = np.mean(dff(analysis_data[W[:, i] >= cut, :]), 0)
    # copy regressors and sort by correlation to data.stimOn - since we expect this to be most prevalent group
    reg_trans = reg_orig.copy()
    c = np.array([np.corrcoef(reg_orig[:, i], data.stimOn)[0, 1] for i in range(n_regs)])
    reg_trans = reg_trans[:, np.argsort(c)[::-1]]
    c = np.sort(c)[::-1]

    with sns.axes_style('whitegrid'):
        fig, ax_on = pl.subplots()
        fig, ax_off = pl.subplots()
        for i in range(n_regs):
            lab = 'Regressor ' + str(i)
            if c[i] > 0:
                ax_on.plot(time, reg_trans[:, i], label=lab)
            else:
                ax_off.plot(time, reg_trans[:, i], label=lab)
        # ax_on.legend(loc=2)
        # ax_off.legend(loc=2)
        ax_on.set_title('ON type regressors')
        ax_on.set_xlabel('Time [s]')
        ax_on.set_ylabel('dF/ F0')
        ax_off.set_title('OFF type regressors')
        ax_off.set_xlabel('Time [s]')
        ax_off.set_ylabel('dF/ F0')

    # create matrix, that for each unit contains its correlation to each regressor as well as to the plane's motor reg
    reg_corr_mat = np.empty((all_activity.shape[0], n_regs+1))
    for i in range(all_activity.shape[0]):
        for j in range(n_regs):
            reg_corr_mat[i, j] = np.corrcoef(all_activity[i, :], reg_trans[:, j])[0, 1]
        reg_corr_mat[i, -1] = np.corrcoef(all_activity[i, :], all_motor[i, :])[0, 1]

    reg_corr_mat[np.abs(reg_corr_mat) < 0.5] = 0
    no_nan = np.sum(np.isnan(reg_corr_mat), 1) == 0
    reg_corr_mat = reg_corr_mat[no_nan, :]

    ab_thresh = np.sum(np.abs(reg_corr_mat) > 0, 1) > 0

    # plot regressor correlation matrix - all units no clustering
    fig, ax = pl.subplots()
    sns.heatmap(reg_corr_mat[np.argsort(ab_thresh)[::-1], :], vmin=-1, vmax=1, center=0, yticklabels=50000)
    ax.set_title('Regressor correlations, thresholded at 0.5')

    # remove all rows that don't have at least one above-threshold correlation
    reg_corr_mat = reg_corr_mat[ab_thresh, :]

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10)
    km.fit(reg_corr_mat)
    # plot sorted by cluster identity
    fig, ax = pl.subplots()
    sns.heatmap(reg_corr_mat[np.argsort(km.labels_), :], vmin=-1, vmax=1, center=0, yticklabels=2500)
    # plot cluster boundaries
    covered = 0
    for i in range(km.n_clusters):
        covered += np.sum(km.labels_ == i)
        ax.plot([0, n_regs+1], [km.labels_.size-covered, km.labels_.size-covered], 'k')
    ax.set_title('Above threshold correlations clustered and sorted')

    # create vector which for all units in all_activity marks their cluster label or -1 if not part of
    # above threshold cluster - note: membership has same size as all originally combined data and should therefore
    # allow to trace back the membership of each experimental timeseries however in order to be used as indexer
    # into all_activity it first need to be reduced to membership[no_nan_aa]
    membership = np.zeros(exp_id.size, dtype=np.float32) - 1
    temp_no_nan_aa = np.zeros(np.sum(no_nan_aa), dtype=np.float32) - 1
    temp_no_nan = np.zeros(np.sum(no_nan), dtype=np.float32) - 1
    temp_no_nan[ab_thresh] = km.labels_
    temp_no_nan_aa[no_nan] = temp_no_nan
    membership[no_nan_aa] = temp_no_nan_aa
    assert membership.size == no_nan_aa.size
    assert membership[no_nan_aa].size == all_activity.shape[0]
    assert membership.size == sum([e.RawData.shape[0] for e in exp_data])

    # determine which fraction of each cluster is made up of the units initially picked for regressor estimations
    dum = discovery_unit_marker[no_nan][ab_thresh]
    cluster_contrib = []
    for i in range(km.n_clusters):
        cluster_contrib.append([np.sum(dum[km.labels_ == i]) / np.sum(km.labels_ == i)])
    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        sns.barplot(data=cluster_contrib, ax=ax)

    # run gram-schmitt process

    def project(u, v):
        """
        Projects the vector v orthogonally onto the line spanned by u
        """
        return np.dot(v, u)/np.dot(u, u)*u

    def gram_schmidt(v, *args):
        """
        Transforms the vector v into a vector that is orthogonal to each vector
        in args and has unit length
        """
        start = v
        for u in args:
            start -= project(u, v)
        return start / np.linalg.norm(start)

    all_u = []
    orthonormals = np.empty_like(reg_trans)
    for i in range(n_regs):
        v = reg_trans[:, i].copy()
        if i == 0:
            orthonormals[:, 0] = v / np.linalg.norm(v)
            all_u.append(orthonormals[:, 0])
        else:
            orthonormals[:, i] = gram_schmidt(v, *all_u)
            all_u.append(orthonormals[:, i])

    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        for i in range(n_regs):
            lab = 'Regressor ' + str(i)
            ax.plot(time, orthonormals[:, i], label=lab)
        ax.legend(loc=2)
        ax.set_title('Stimulus based orthogonal regressors')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('dF/ F0')

    # perform linear regression using the stimulus regressors
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(orthonormals, all_activity.T)
    # compute residuals and mean-distance - re-transpose to original shape
    residuals = (all_activity.T - (np.dot(orthonormals, lr.coef_.T) + lr.intercept_)).T
    mean_dist = all_activity - np.mean(all_activity, 1, keepdims=True)
    # compute R2 values
    r2_sensory_fit = 1 - np.sum(residuals**2, 1) / np.sum(mean_dist**2, 1)
    del residuals
    del mean_dist
    # recompute regressions unit-by-unit (should be done plane-by-plane for efficiency!!!) with motor regressors
    # NOTE: As null distribution maybe a version with shuffled (wrong-plane) motor regressors could be computed!
    r2_sensory_motor_fit = []
    for i in range(all_activity.shape[0]):
        mot_reg = gram_schmidt(all_motor[i, :].copy(), *all_u)
        regs = np.c_[orthonormals, mot_reg]
        if np.any(np.isnan(regs)):
            r2_sensory_motor_fit.append(0)
        else:
            lr.fit(regs, all_activity[i, :])
            r2_sensory_motor_fit.append(lr.score(regs, all_activity[i, :]))
    r2_sensory_motor_fit = np.array(r2_sensory_motor_fit)

    r2_sensory_motor_shuffle = []
    for i in range(all_activity.shape[0]):
        shift = np.random.randint(50000, 100000)
        pick = (i + shift) % all_motor.shape[0]
        mot_reg = gram_schmidt(all_motor[pick, :].copy(), *all_u)
        regs = np.c_[orthonormals, mot_reg]
        if np.any(np.isnan(regs)):
            r2_sensory_motor_shuffle.append(0)
        else:
            lr.fit(regs, all_activity[i, :])
            r2_sensory_motor_shuffle.append(lr.score(regs, all_activity[i, :]))

    r2_sensory_motor_shuffle = np.array(r2_sensory_motor_shuffle)

    # compute confidence band based on shuffle
    ci = 99
    bedges = np.linspace(0, 1, 51)
    bc = bedges[:-1] + np.diff(bedges)/2
    conf = np.zeros(bc.size)
    for i in range(bedges.size - 1):
        all_in = np.logical_and(r2_sensory_fit >= bedges[i], r2_sensory_fit < bedges[i + 1])
        if np.sum(all_in > 0):
            conf[i] = np.percentile(r2_sensory_motor_shuffle[all_in], ci)
        else:
            conf[i] = np.nan

    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        ax.scatter(r2_sensory_fit, r2_sensory_motor_shuffle, alpha=0.4, color='r', s=3)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$R^2$ Sensory regression')
        ax.set_ylabel('$R^2$ Sensory + Motor regression')
        ax.set_title('Shuffled motor regressor control')

    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        ax.scatter(r2_sensory_fit, r2_sensory_motor_fit, alpha=0.4, s=3)
        ax.plot([0, 1], [0, 1], 'k--')
        # plot confidence band
        ax.fill_between(bc, bc, conf, color='m', alpha=0.5)
        ax.plot(bc, conf, 'm')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$R^2$ Sensory regression')
        ax.set_ylabel('$R^2$ Sensory + Motor regression')
        ax.set_title('Boost of fit by including motor regressor')

    # at various timeshifts of the motor regressor compute pure motor r2
    t_shifts = [0, 1, 2, 5, 10, 20]
    motor_shift_r2 = np.zeros((len(t_shifts), all_motor.shape[0]))
    for i, ts in enumerate(t_shifts):
        shifted = np.roll(all_motor, ts, 1)
        for j in range(all_motor.shape[0]):
            c = np.corrcoef(all_activity[j, :], shifted[j, :])[0, 1]**2
            if not np.isnan(c):
                motor_shift_r2[i, j] = c

    r2_bins = np.linspace(0, 1, 20)
    r2bc = r2_bins[:-1] + np.diff(r2_bins)/2
    motor_shift_h = np.vstack([np.histogram(row, r2_bins)[0] for row in motor_shift_r2])
    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        for i, ts in enumerate(t_shifts):
            ax.plot(r2bc, motor_shift_h[i, :], label='Shift of ' + str(ts/5) + ' s')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('$R^2$ of motor correlation')
        ax.set_ylabel('Count')

    #REMOVE THE FOLLOWING LATER
    eid = exp_id[no_nan_aa]
    eid2 = eid[no_nan][ab_thresh]
    fb = eid2 < 3
    mhb = np.logical_and(eid2 > 2, eid2 < 8)
    trig = eid2 > 7
    region_mat = np.zeros((10, 3))
    for i in range(10):
        region_mat[i, 0] = np.sum(fb[km.labels_ == i])
        region_mat[i, 1] = np.sum(mhb[km.labels_ == i])
        region_mat[i, 2] = np.sum(trig[km.labels_ == i])
    rm_norm = region_mat / np.sum(region_mat, 0, keepdims=True)
    rm_norm = rm_norm / np.sum(rm_norm, 1, keepdims=True)
    pl.figure()
    sns.heatmap(rm_norm, cmap='bone_r', xticklabels=['FB', 'MB-HB', 'TG'])
    pl.title('Contribution of "Brain regions" to clusters')

