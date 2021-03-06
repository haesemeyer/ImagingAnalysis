from mh_2P import OpenStack, TailData, UiGetFile, NucGraph, CorrelationGraph, SOORepeatExperiment, SLHRepeatExperiment
from mh_2P import MakeNrrdHeader, TailDataDict, vec_mat_corr, KDTree, MotorContainer
from motorPredicates import left_bias_bouts, right_bias_bouts, unbiased_bouts, high_bias_bouts
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from time import perf_counter
import scipy.signal as signal
import pickle
import nrrd
import sys
import os
import subprocess as sp
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')
from mhba_basic import Crosscorrelation


class max_cluster:
    def __init__(self, max_index):
        self.labels_ = max_index
        self.n_clusters = np.unique(self.labels_).size


def dff(fluomat, n_pre=5*60):
    f0 = np.mean(fluomat[:, 15:n_pre-15], 1, keepdims=True)
    f0[f0 == 0] = 0.1
    return(fluomat-f0)/f0


def dist2next1(v):
    """
    Given v returns a vector of the same size
    in which each 1 in v has been replaced by
    the smallest element-wise distance to the
    next neighboring 1
    """
    if type(v) is list:
        v = np.array(v)
    out = np.zeros(v.size, dtype=np.float32)
    if np.sum(v) == 0:
        return out
    # get indices of each 1 in the array
    ix1 = np.arange(v.size)[v > 0]
    # for each one compute distances to neighboring 1s
    d1 = np.r_[np.inf, np.diff(ix1), np.inf]
    min_d1 = np.empty(ix1.size, dtype=np.float32)
    for i in range(min_d1.size):
        min_d1[i] = np.min(d1[i:i+2])
    out[v > 0] = min_d1
    return out


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
    if corr_mat.ndim == 2:
        # matrix input
        return np.sum(corr_mat**2 > r2_thresh, 1) - 1
    else:
        # 1D vector
        return np.sum(corr_mat**2 > r2_thresh) - 1


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
    if corr_mat.ndim == 2:
        # matrix input
        return np.array([np.unique(exp_ids[above]).size-1 for above in corr_above])
    else:
        # 1D vector
        return np.unique(exp_ids[corr_above]).size-1


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
        clusterNumber: The cluster number(s) for which to create stack. If a list all ROI's that belong to any of the
        list's clusters will be marked
    """
    assert len(experiment_data.graph_info) == unit_cluster_ids.size
    selector = np.zeros(unit_cluster_ids.size)
    id_string = ""
    try:
        for cn in clusterNumber:
            selector = np.logical_or(selector, unit_cluster_ids == cn)
            id_string = id_string + "_" + str(cn)
    except TypeError:
        selector = unit_cluster_ids == clusterNumber
        id_string = str(clusterNumber)
    zstack = MakeMaskStack(experiment_data, selector.astype(np.float32),
                           np.zeros(unit_cluster_ids.size), np.zeros(unit_cluster_ids.size), 0.5, 1.0)
    # recode and reformat zstack for nrrd saving
    nrrdStack = np.zeros((zstack.shape[2], zstack.shape[1], zstack.shape[0]), dtype=np.uint8, order='F')
    for i in range(zstack.shape[0]):
        nrrdStack[:, :, i] = (zstack[i, :, :, 0]*255).astype(np.uint8).T
    header = MakeNrrdHeader(nrrdStack, 500/512/exp_zoom_factor)
    assert np.isfortran(nrrdStack)
    out_name = GetExperimentBaseName(experiment_data) + '_C' + id_string + '.nrrd'
    nrrd.write(out_name, nrrdStack, header)
    return out_name


def ReformatROIStack(stackFilename, referenceFolder="E:/Dropbox/ReferenceBrainCreation/"):
    nameOnly = os.path.basename(stackFilename)
    cluster_id_start = nameOnly.find("_C")
    ext_start = nameOnly.lower().find(".nrrd")
    transform_name = nameOnly[:cluster_id_start]
    transform_file = referenceFolder + transform_name + '/' + transform_name + '_ffd5.xform'
    assert os.path.exists(transform_file)
    referenceBrain = referenceFolder + 'H2BGc6s_Reference_8.nrrd'
    outfile = referenceFolder + nameOnly[:ext_start] + '_reg.nrrd'
    command = 'reformatx --outfile '+outfile+' --floating '+stackFilename+' '+referenceBrain+' '+transform_file
    sp.run(command)


def ReformatTGROIStack(stackFilename, leftTG=True, referenceFolder="E:/Dropbox/ReferenceTrigeminalCreation/"):
    assert "Trigem" in stackFilename
    nameOnly = os.path.basename(stackFilename)
    trigem_start = nameOnly.find("Trigem")
    fnum_start = trigem_start+7
    ext_start = nameOnly.lower().find(".nrrd")
    transform_name = nameOnly[fnum_start:fnum_start+2]
    if leftTG:
        transform_name += "_into_left"
        referenceBrain = referenceFolder + "Left_04_05_09_11.nrrd"
    else:
        transform_name += "_into_right"
        referenceBrain = referenceFolder + "Right_06_07_08_10.nrrd"
    transform_file = referenceFolder + transform_name + '/ffd5.xform'
    assert os.path.exists(transform_file)
    assert os.path.exists(referenceBrain)
    outfile = referenceFolder + nameOnly[:ext_start] + '_reg.nrrd'
    command = 'reformatx --outfile ' + outfile + ' --floating ' + stackFilename + ' ' + referenceBrain + ' ' + transform_file
    sp.run(command)


def MakeMeanNrrd():
    sourceFiles = UiGetFile([('Nrrd files', '.nrrd')], True)
    header = None
    mean_stack = None
    for f in sourceFiles:
        if header is None:
            data, header = nrrd.read(f)
            data[data > 0] = 1
            mean_stack = data.copy().astype(np.float32)
        else:
            data = nrrd.read(f)[0]
            data[data > 0] = 1
            mean_stack += data
    mean_stack /= len(sourceFiles)
    mean_stack *= 255
    mean_stack[mean_stack > 255] = 255
    directory = os.path.dirname(sourceFiles[0]) + '/'
    nrrd.write(directory+"mean_stack.nrrd", mean_stack.astype(np.uint8), header)


def CreateAndReformatAllClusterStacks(membership):
    cluster_ids = np.unique(membership[membership != -1])
    for i, e in enumerate(exp_data):
        if "FB" in e.graph_info[0][0]:
            zf = 2
        elif "MidHB" in e.graph_info[0][0]:
            zf = 1
        else:
            print("Non forebrain or mid-hindbrain stacks currently not processed")
            continue
        sfiles = [MakeAndSaveROIStack(e, zf, membership[exp_id == i], c) for c in cluster_ids]
        for s in sfiles:
            ReformatROIStack(s)
    return True


def is_left_tg(name):
    """
    Returns true if name comes from a left trigeminal tiff or fals if from right
    """
    if "_04_Z" in name or "_05_Z" in name or "_09_Z" in name or "_11_Z" in name or "_12_Z" in name:
        return True
    else:
        return False


def CreateAndReformatAllTGClusterStacks(membership):
    cluster_ids = np.unique(membership[membership != -1])
    for i, e in enumerate(exp_data):
        name = e.graph_info[0][0]
        if "Trigem" in name:
            sfiles = [MakeAndSaveROIStack(e, 2, membership[exp_id == i], c) for c in cluster_ids]
            for s in sfiles:
                ReformatTGROIStack(s, is_left_tg(name))
        else:
            continue


def DumpAnalysisDataHdf5(filename):
    """
    Save all created analysis data into one hdf5 file for easy re-analysis
    Args:
        filename: The name of the file to create - if file already exists exception will be raised
    Returns:
        True on success
    """
    import h5py
    try:
        dfile = h5py.File(filename, 'w-')  # fail if file already exists
    except OSError:
        print("Could not create file. Make sure a file with the same name doesn't exist in this location.")
        return False
    try:
        # save major data structures compressed
        dfile.create_dataset("all_activity", data=all_activity, compression="gzip", compression_opts=9)
        dfile.create_dataset("avg_analysis_data", data=avg_analysis_data, compression="gzip", compression_opts=9)
        # pickle our experiment data as a byte-stream
        pstring = pickle.dumps(exp_data)
        dfile.create_dataset("exp_data_pickle", data=np.void(pstring))
        # pickle the tail-data dict of mc_all - for some reason we cannot pickle the actual object without running
        # out of memory so instead we save individual tail-data files with keys corresponding to the .tail files
        for i, k in enumerate(mc_all.tdd.fileNames):
            pstring = pickle.dumps(mc_all.tdd[k])
            dfile.create_dataset(k, data=np.void(pstring))
        # save smaller arrays uncompressed
        dfile.create_dataset("exp_id", data=exp_id)
        dfile.create_dataset("membership", data=membership)
        dfile.create_dataset("no_nan", data=no_nan)
        dfile.create_dataset("no_nan_aa", data=no_nan_aa)
        dfile.create_dataset("reg_corr_mat", data=reg_corr_mat)
        dfile.create_dataset("reg_trans", data=reg_trans)
        # dfile.create_dataset("r2_sensory_fit", data=r2_sensory_fit)
        # dfile.create_dataset("r2_sensory_motor_fit", data=r2_sensory_motor_fit)
        # dfile.create_dataset("r2_sensor_motor_shuffle", data=r2_sensory_motor_shuffle)
        # save transformed nuclear centroids if they exist, otherwise create them
        try:
            tf_centroids
        except NameError:
            print("Transforming all coordinate centroids")
            tf_centroids = np.vstack([transform_centroid_coordinates(save_nuclear_coordinates(e)) for e in exp_data])
        dfile.create_dataset("tf_centroids", data=tf_centroids)
    finally:
        dfile.close()
    return True


def expVigor(expData):
    done = dict()
    vigs = []
    for i in range(expData.Vigor.shape[0]):
        if expData.graph_info[i][0] in done:
            continue
        done[expData.graph_info[i][0]] = True
        vigs.append(expData.Vigor[i, :])
    return np.vstack(vigs)


def printElapsed():
    elapsed = perf_counter() - t_start
    print(str(elapsed) + " s elapsed since start.", flush=True)


def stack_coordinate_centroid(gi):
    """
    Returns an (x,y,z) centroid of a graph in (um) coordinates (registration space)
    Args:
        gi: The graph information for a particular nucleus

    Returns:
        A 3 element numpy array with the x y and z coordinate of the nucleus center in microns
    """
    def scale():
        if "FB" in gi[0]:
            return 500/512/2
        elif "MidHB" in gi[0]:
            return 500/512
        elif "Trigem" in gi[0]:
            return 500/512/2
        else:
            raise ValueError("Could not determine zoom for stack")
    x1 = gi[0].find('_Z_') + 3
    x2 = gi[0].find('_0.')
    verts = gi[1]
    zcoord = float(gi[0][x1:x2]) * 2.5
    sc = scale()
    return np.array([np.mean([v[1] for v in verts]) * sc, np.mean([v[0] for v in verts]) * sc, zcoord])


def save_nuclear_coordinates(experiment):
    """
    Saves all nuclear centroids for a given experiment
    Returns:
        The path and filename where coordinates were saved
    """
    out_name = GetExperimentBaseName(experiment) + "_nucCentroids" + '.txt'
    try:
        all_centroids = np.vstack([stack_coordinate_centroid(gi) for gi in experiment.graph_info])
    except ValueError:
        print("Did not recognize experimental region coordinates are invalid")
        all_centroids = np.zeros((len(experiment.graph_info), 3))
    np.savetxt(out_name, all_centroids, fmt='%.1f')
    return out_name


def transform_centroid_coordinates(centroidFilename, referenceFolders=("E:/Dropbox/ReferenceBrainCreation/",
                                                                       "E:/Dropbox/ReferenceTrigeminalCreation/")):
    """
    Transform centroid coordinates from the specified file into the reference brain space and return
    an array of those coordinates
    Args:
        centroidFilename: The file to convert. NOTE: The filename determines which transformation to apply!
        referenceFolders: Tuple with the paths of the main and trigeminal reference brain folders

    Returns:
        A matrix of the transformed points
    """
    if "Trigem" in centroidFilename:
        referenceFolder = referenceFolders[1]
    else:
        referenceFolder = referenceFolders[0]
    nameOnly = os.path.basename(centroidFilename)
    suffix_start = nameOnly.find("_nucCentroids")
    ext_start = nameOnly.lower().find(".txt")
    transform_name = nameOnly[:suffix_start]
    if "Trigem" in centroidFilename:
        # unfortunately trigeminal registrations don't follow the most useful naming scheme
        # determine whether this is left of right
        zstart = centroidFilename.find("_nuc")
        snippet = centroidFilename[zstart-2:zstart]
        if "04" in snippet or "05" in snippet or "09" in snippet or "11" in snippet or "12" in snippet:
            suffix = "_into_left"
        else:
            suffix = "_into_right"
        folder_name = snippet + suffix
        transform_file = referenceFolder + folder_name + "/ffd5.xform"
    else:
        transform_file = referenceFolder + transform_name + '/' + transform_name + '_ffd5.xform'
    if not os.path.exists(transform_file):
        print("Could not determine transform file returning all nan ({0})".format(transform_file))
        outshape = np.genfromtxt(centroidFilename).shape
        return np.full(outshape, np.nan)
    outfile = referenceFolder + nameOnly[:ext_start] + '_transform.txt'
    command = 'cat ' + centroidFilename + ' | ' + 'streamxform --separator "," -- --inverse ' + transform_file + ' > ' + outfile
    sp.run(command, shell=True)  # shell=True is required for the piping to work!!
    transformed = np.genfromtxt(outfile, delimiter=',')[:, :3]
    # whenever there is FAILED in the file there will be a nan value in that column
    # nan-out every row that has at least one NaN
    has_nan = np.sum(np.isnan(transformed), 1) > 0
    transformed[has_nan, :] = np.nan
    return transformed


def create_centroid_stack(centroids_um, stack_type="MAIN", brightness=1):
    """
    Create a (refernce) stack with given centroids marked as single dots
    Args:
        centroids_um: The (x,y,z) centroid coordinates in um
        stack_type: Either "MAIN" or "TG" to obtain overall dimensions and resolution
        brightness: Either a scalar btw. 0 and 1 defining the brightness of all points or array with value per centroid

    Returns:
        [0]: Stack
        [1]: Compatible NRRD header
    """
    if not np.isscalar(brightness):
        if brightness.size != centroids_um.shape[0]:
            raise ValueError("Brightness either needs to be scaler of have one element per centroid")
    res_z = 2.5  # all our stacks have 2.5 um z-resolution
    if stack_type == "MAIN":
        res_xy = 500/512/1.5
        shape = (770, 1380, 100)
    elif stack_type == "TG":
        res_xy = 500/512/2
        shape = (512, 512, 30)
    else:
        raise ValueError("stack_type has to be one of 'MAIN' or 'TG'")
    stack = np.zeros(shape, dtype=np.uint8)
    header = MakeNrrdHeader(stack, res_xy, res_z)
    for i, cents in enumerate(centroids_um):
        if np.any(np.isnan(cents)):
            continue
        if np.isscalar(brightness):
            b = int(255 * brightness)
        else:
            b = int(255 * brightness[i])
        stack[int(cents[0] / res_xy), int(cents[1] / res_xy), int(cents[2] / res_z)] = b
    return stack, header


def get_stack_types(exp_data):
    """
    Function to determine based on filename whether a given cell belongs to a brain, left or right TG stack
    Args:
        exp_data: The experiment data structures with the cells in question

    Returns:
        len(exp_data) sized array of strings with either "MAIN", "TG_LEFT" or "TG_RIGHT" identifying the stack type
    """
    tf_names = [g[0] for e in exp_data for g in e.graph_info]
    types = []
    for i, name in enumerate(tf_names):
        if "Trigem" in name:
            if is_left_tg(name):
                types.append("TG_LEFT")
            else:
                types.append("TG_RIGHT")
        else:
            types.append("MAIN")
    return np.array(types)


def rem_nan_1d(x: np.ndarray) -> np.ndarray:
    """
    Remove all NaN values from given vector
    """
    return x.copy()[np.logical_not(np.isnan(x))]


def all_pw_dist(coords: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise distances with self-distances set to nan
    Args:
        coords: Array of coordinates for which to compute pairwise distances

    Returns:
        Matrix of size coords.shape[0]xcoords.shape[0] with all pairwise distances
    """
    distmat = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(distmat.shape[0]):
        d = np.sum((coords[i, :][None, :] - coords)**2, 1)
        assert np.isnan(d[i]) or d[i] == 0
        d[i] = np.nan
        distmat[i, :] = np.sqrt(d)
    return distmat


def min_dist(points: np.ndarray, partners: np.ndarray, allow_0=False, avgSmallest=1) -> np.ndarray:
    """
    Computes for each point the distance to the closest partner
    Args:
        points: The points (nx3 array) for which to compute the minimal partner distance
        partners: The partners to which distances will be computed for each point
        allow_0: If true, a 0 distance won't be converted to NaN
        avgSmallest: If larger than 1 instead of computing the minimal distance compute avg. across n smallest

    Returns:
        n long vector with the distance to the closest partner for each point
    """
    if avgSmallest < 1:
        ValueError("avgSmallest can't be smaller 1")
    # for larger set of partner points use k-d-tree for faster computation
    # cut-off has been determined empirically
    if partners.shape[0] >= 20000:
        kd_tree = KDTree(partners)
        return kd_tree.avg_min_distances(points, avgSmallest, allow_0)

    mds = np.zeros(points.shape[0])
    for i, p in enumerate(points):
        d = np.sqrt(np.sum((p[None, :] - partners)**2, 1))
        if avgSmallest == 1:
            try:
                if allow_0:
                    mds[i] = np.nanmin(d)
                else:
                    mds[i] = np.min(d[d > 0])
            except ValueError:
                mds[i] = np.nan
        else:
            # need to remove nan values, sort and then pick n smallest
            if allow_0:
                d = rem_nan_1d(d)
            else:
                d = d[d > 0]
            if d.size == 0:
                mds[i] = np.nan
            elif d.size <= avgSmallest:
                mds[i] = np.mean(d)
            else:
                d = np.sort(d)
                mds[i] = np.mean(d[:avgSmallest])
    return mds


def create_updater(kept: np.ndarray):
    """
    Creates a function to update indices such that a given index into an element removed array will point to the
    corresponding element in a full array before removal
    Args:
        kept: Indicates the elements kept from the original array using non-zero values
    Returns:
        A function for updating indices according to kept
    """
    def upd_ix(ix):
        nonlocal mapping
        if ix < 0 or ix >= mapping.size:
            raise ValueError("Index is out of bounds for element removed array")
        return int(mapping[ix])
    mapping = np.zeros(np.sum(kept != 0), dtype=int)
    skipped = 0
    for i in range(kept.size):
        if kept[i] != 0:
            mapping[i - skipped] = i
        else:
            skipped += 1
    return upd_ix


ani = None
ani_stack = []
ani_im = []
ani_show_ix = 0
ani_fig = None


def updatefig(*args):
    global ani_stack, ani_show_ix, ani_im
    ani_show_ix = (ani_show_ix + 1) % (ani_stack.shape[0] - 1)
    ani_im.set_array(ani_stack[ani_show_ix, :, :])
    return ani_im,


def animate_ROI(roi_index):
    pl.ioff()
    from scipy.signal import lfilter
    import matplotlib.animation as animation
    global ani_stack, ani_im, ani_show_ix, ani_fig, ani
    exp_index = exp_id[roi_index].astype(int)
    all_idx = np.arange(exp_id.size).astype(int)
    in_exp_idx = all_idx[roi_index] - np.min(all_idx[exp_id == exp_index])
    source = exp_data[exp_index].graph_info[in_exp_idx][0]
    ani_stack = np.load(source[:-4] + "_stack.npy").astype(np.float32)
    verts = exp_data[exp_index].graph_info[in_exp_idx][1]
    minx = min([v[0] for v in verts]) - 50
    maxx = max([v[0] for v in verts]) + 50
    miny = min([v[1] for v in verts]) - 50
    maxy = max([v[1] for v in verts]) + 50
    if minx < 0:
        minx = 0
    if miny < 0:
        miny = 0
    if maxx > ani_stack.shape[1]:
        maxx = ani_stack.shape[1]
    if maxy > ani_stack.shape[2]:
        maxy = ani_stack.shape[2]
    ani_stack = ani_stack[:, minx:maxx, miny:maxy].copy()
    ani_stack = lfilter(np.ones(10)/10, 1, ani_stack, 0)
    ani_show_ix = 0
    ani_fig = pl.figure()
    ani_im = pl.imshow(ani_stack[0, :, :], cmap=pl.get_cmap("bone"), animated=True)

    ani = animation.FuncAnimation(ani_fig, updatefig, interval=50, blit=True)
    pl.show()


if __name__ == "__main__":
    print("Load data files", flush=True)
    exp_data = []
    dfnames = UiGetFile([('Experiment data', '.pickle')], True)
    t_start = perf_counter()
    for name in dfnames:
        f = open(name, 'rb')
        d = pickle.load(f)
        exp_data.append(d)
    is_pot_stim = np.array([], dtype=np.bool)  # for each unit whether it is potentially a stimulus driven unit
    exp_id = np.array([])  # for each unit the experiment which it came from
    stim_phase = np.array([])  # for each unit the phase at stimulus frequency during the sine-presentation
    all_activity = np.array([])
    source_files = []  # for each unit the imaging file from which it was derived
    for i, data in enumerate(exp_data):
        m_corr = data.motorCorrelation(0)[0].ravel()
        stim_fluct = data.computeStimulusEffect(0)[0].ravel()
        exp_id = np.r_[exp_id, np.full(m_corr.size, i, np.int32)]
        ips = stim_fluct >= 1
        ips = np.logical_and(ips, m_corr < 0.4)
        is_pot_stim = np.r_[is_pot_stim, ips]
        if i == 0:
            # NOTE: RawData field is 64 bit float - convert to 32 when storing in all_activity
            all_activity = data.RawData.astype(np.float32)
        else:
            all_activity = np.r_[all_activity, data.RawData.astype(np.float32)]
        # after this point, the raw-data and vigor fields of experiment data are unecessary
        data.RawData = None
        data.Vigor = None
        source_files += [g[0] for g in data.graph_info]

    # shuffle all_activity - trials get shuffled individually
    # rolls = np.zeros(all_activity.shape[0], dtype=np.int16)
    # tlen = all_activity.shape[1] // 3
    # for i, row in enumerate(all_activity):
    #     for tnum in range(3):
    #         tact = row[tlen*tnum: tlen*(tnum+1)]
    #         r = np.random.randint(0, tlen)
    #         shuff = np.roll(tact, r)
    #         row[tlen * tnum: tlen * (tnum + 1)] = shuff
    #     all_activity[i, :] = row

    # create our motor event containers
    i_time = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    tc = exp_data[0].caTimeConstant
    mc_all = MotorContainer(source_files, i_time, tc)
    mc_left = MotorContainer(source_files, i_time, tc, predicate=left_bias_bouts, tdd=mc_all.tdd)
    mc_right = MotorContainer(source_files, i_time, tc, predicate=right_bias_bouts, tdd=mc_all.tdd)
    mc_straight = MotorContainer(source_files, i_time, tc, predicate=unbiased_bouts, tdd=mc_all.tdd)
    mc = [mc_all, mc_left, mc_right, mc_straight]

    print("Data loaded and aggregated", flush=True)
    printElapsed()
    # Filter our activity traces with a gaussian window for rate estimation - sigma = 0.3 seconds
    window_size = 0.3 * exp_data[0].frameRate
    all_activity = gaussian_filter1d(all_activity, window_size, axis=1)
    # Plot the frequency gain characteristics of the filter used
    delta = np.zeros(21)
    delta[10] = 1
    filter_coeff = gaussian_filter1d(delta, window_size)
    nyquist = exp_data[0].frameRate/2
    w, h = signal.freqz(filter_coeff, worN=8000)
    with sns.axes_style('whitegrid'):
        pl.figure()
        pl.plot((w/np.pi)*nyquist, np.absolute(h), linewidth=2)
        pl.xlabel('Frequency [Hz]')
        pl.ylabel('Gain')
        pl.title('Frequency response of gaussian filter')
        pl.ylim(-0.05, 1.05)
    print("Activity filtered", flush=True)
    printElapsed()

    # for each cell that passes is_pot_stim holds the count of the number of other cells and other experiments that
    # are above the correlation threshold
    n_cells_above = np.zeros(is_pot_stim.sum(), dtype=np.int32)
    n_exp_above = np.zeros(is_pot_stim.sum(), dtype=np.int32)

    filter_r2_thresh = 0.4

    # make a temporary copy of all stimulus relevant units
    stim_act = all_activity[is_pot_stim, :].copy()
    rowmean = np.mean(stim_act, 1, keepdims=True)
    stim_act -= rowmean
    norms = np.linalg.norm(stim_act, axis=1)

    for i, row in enumerate(all_activity[is_pot_stim, :]):
        corrs = vec_mat_corr(row, stim_act, False, vnorm=norms[i], mnorm=norms)
        n_cells_above[i] = n_r2_above_thresh(corrs, filter_r2_thresh)
        n_exp_above[i] = n_exp_r2_above_thresh(corrs, filter_r2_thresh, exp_id[is_pot_stim])

    del stim_act

    # generate plot of number of cells to analyze for different "other cell" and "n experiment" criteria
    n_exp_to_test = [0, 1, 2, 3, 4, 5]
    n_cells_to_test = list(range(30))
    exp_above = np.zeros((len(n_exp_to_test), n_cells_above.size))
    cells_above = np.zeros((len(n_cells_to_test), n_cells_above.size))
    for i, net in enumerate(n_exp_to_test):
        exp_above[i, :] = n_exp_above >= net
    for i, nct in enumerate(n_cells_to_test):
        cells_above[i, :] = n_cells_above >= nct

    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        for i, net in enumerate(n_exp_to_test):
            n_remain = []
            for j, nct in enumerate(n_cells_to_test):
                n_remain.append(np.sum(np.logical_and(exp_above[i, :], cells_above[j, :])) / n_cells_above.size)
            ax.plot(n_cells_to_test, n_remain, 'o', label="At least "+str(net)+" other fish")
        ax.set_xlabel('Number of other cells with $R^2$ > ' + str(filter_r2_thresh))
        ax.set_ylabel('Fraction of stimulus cells to analyze')
        ax.set_xlim(-0.2)
        ax.set_yscale('log')
        ax.legend()

    # get all cells that have at least 20 other cells with a timeseries R2>0.5 and are spread
    # across at least 3 experiments
    exp_g_1 = n_exp_above > 2
    c_g_9 = n_cells_above > 19
    to_analyze = np.logical_and(exp_g_1, c_g_9)
    analysis_data = all_activity[is_pot_stim, :][to_analyze, :]
    # marks, which of all units where used to derive regressors
    discovery_unit_marker = np.zeros(all_activity.shape[0], dtype=bool)
    discovery_unit_marker[is_pot_stim] = to_analyze

    # remove NaN containing traces from activity and motor matrix
    no_nan_aa = np.sum(np.logical_or(np.isnan(all_activity), np.isinf(all_activity)), 1) == 0
    all_activity = all_activity[no_nan_aa, :]
    # since our MotorContainer objects are not restricted to elements identified by no_nan_aa we need to create a
    # function that allows us to update indices into a no_nan_aa restricted array accordingly
    update_index = create_updater(no_nan_aa)
    discovery_unit_marker = discovery_unit_marker[no_nan_aa]
    print("Data filtering complete", flush=True)
    printElapsed()

    # analyze per-experiment swim vigors
    # all_expVigors = np.vstack([np.mean(expVigor(data), 0) for data in exp_data])
    # all_expVigors = all_expVigors / np.mean(all_expVigors, 1, keepdims=True)

    # TODO: The following is experiment specific - needs cleanup!
    avg_analysis_data = np.zeros((analysis_data.shape[0], analysis_data.shape[1] // 3))
    avg_analysis_data += analysis_data[:, :825]
    avg_analysis_data += analysis_data[:, 825:825*2]
    avg_analysis_data += analysis_data[:, 825*2:]
    # use spectral clustering to identify our regressors. Use thresholded correlations as the affinity
    aad_corrs = np.corrcoef(avg_analysis_data)
    aad_corrs[aad_corrs < 0.26] = 0  # 99th percentile of correlations of a *column* shuffle of avg_analysis_data
    n_regs = 6  # extract 6 clusters as regressors
    spec_embed = SpectralEmbedding(n_components=3, affinity='precomputed')
    spec_clust = SpectralClustering(n_clusters=n_regs, affinity='precomputed')
    spec_embed_coords = spec_embed.fit_transform(aad_corrs)
    spec_clust_ids = spec_clust.fit_predict(aad_corrs)

    reg_orig = np.empty((analysis_data.shape[1], n_regs))
    time = np.arange(reg_orig.shape[0]) / 5
    for i in range(n_regs):
        reg_orig[:, i] = np.mean(dff(analysis_data[spec_clust_ids == i, :]), 0)
    # copy regressors and sort by correlation to data.stimOn - since we expect this to be most prevalent group
    reg_trans = reg_orig.copy()
    c = np.array([np.corrcoef(reg_orig[:, i], data.stimOn)[0, 1] for i in range(n_regs)])

    # make 3D plot of clusters in embedded space
    n_on = np.sum(c > 0)
    n_off = n_regs - n_on
    cols_on = sns.palettes.color_palette('bright', n_on)
    cols_off = sns.palettes.color_palette('deep', n_off)
    count_on = 0
    count_off = 0
    with sns.axes_style('white'):
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n_regs):
            if c[i] > 0:
                ax.scatter(spec_embed_coords[spec_clust_ids == i, 0], spec_embed_coords[spec_clust_ids == i, 1],
                           spec_embed_coords[spec_clust_ids == i, 2], s=5, c=cols_on[count_on])
                count_on += 1
            else:
                ax.scatter(spec_embed_coords[spec_clust_ids == i, 0], spec_embed_coords[spec_clust_ids == i, 1],
                           spec_embed_coords[spec_clust_ids == i, 2], s=5, c=cols_off[count_off])
                count_off += 1

    reg_trans = reg_trans[:, np.argsort(c)[::-1]]
    c = np.sort(c)[::-1]

    # plot our on and off type clusters in two different plots
    with sns.axes_style('whitegrid'):
        fig, ax_on = pl.subplots()
        fig, ax_off = pl.subplots()
        for i in range(n_regs):
            lab = 'Regressor ' + str(i)
            if c[i] > 0:
                ax_on.plot(time, reg_trans[:, i], label=lab, c=cols_on[i])
            else:
                ax_off.plot(time, reg_trans[:, i], label=lab, c=cols_off[i-n_on])
        # ax_on.legend(loc=2)
        # ax_off.legend(loc=2)
        ax_on.set_title('ON type regressors')
        ax_on.set_xlabel('Time [s]')
        ax_on.set_ylabel('dF/ F0')
        ax_off.set_title('OFF type regressors')
        ax_off.set_xlabel('Time [s]')
        ax_off.set_ylabel('dF/ F0')
    print("Stimulus regressor derivation complete", flush=True)
    printElapsed()

    # create matrix, that for each unit contains its correlation to each regressor as well as to the motor regs
    reg_corr_mat = np.empty((all_activity.shape[0], n_regs+len(mc)), dtype=np.float32)
    for i in range(all_activity.shape[0]):
        for j in range(n_regs):
            reg_corr_mat[i, j] = np.corrcoef(all_activity[i, :], reg_trans[:, j])[0, 1]
        for j in range(len(mc)):
            # for the motor regressors we need to take care of updating our index
            reg_corr_mat[i, n_regs + j] = np.corrcoef(all_activity[i, :], mc[j][update_index(i), :])[0, 1]

    reg_corr_th = 0.6
    reg_corr_mat[np.abs(reg_corr_mat) < reg_corr_th] = 0
    # exclude cells where correlations with either sensory regressor are NaN *but not* cells for which one of the
    # motor-regressors might be NaN (as we don't expect all motor types to be executed in each plane)
    no_nan = np.sum(np.isnan(reg_corr_mat[:, :n_regs + 1]), 1) == 0
    reg_corr_mat = reg_corr_mat[no_nan, :]

    ab_thresh = np.sum(reg_corr_mat > 0, 1) > 0

    # plot regressor correlation matrix - all units no clustering
    fig, ax = pl.subplots()
    sns.heatmap(reg_corr_mat[np.argsort(ab_thresh)[::-1], :], vmin=-1, vmax=1, center=0, yticklabels=75000)
    ax.set_title('Regressor correlations, thresholded at ' + str(reg_corr_th))

    # remove all rows that don't have at least one above-threshold correlation
    # NOTE: The copy statement below is required to prevent a stale copy of the full-sized array to remain in memory
    reg_corr_mat = reg_corr_mat[ab_thresh, :].copy()

    km = max_cluster(np.nanargmax(reg_corr_mat, 1))
    # km.fit(reg_corr_mat)
    # plot sorted by cluster identity
    fig, ax = pl.subplots()
    sns.heatmap(reg_corr_mat[np.argsort(km.labels_), :], vmin=-1, vmax=1, center=0, yticklabels=5000)
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
    assert membership.size == sum([len(e.graph_info) for e in exp_data])

    # determine which fraction of each cluster is made up of the units initially picked for regressor estimations
    dum = discovery_unit_marker[no_nan][ab_thresh]
    cluster_contrib = []
    for i in range(km.n_clusters):
        cluster_contrib.append([np.sum(dum[km.labels_ == i]) / np.sum(km.labels_ == i)])
    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        sns.barplot(data=cluster_contrib, ax=ax)
        ax.set_xlabel('Cluster #')
        ax.set_ylabel('Cluster fraction from discovery units')
        ax.set_ylim(0, 1)
    print("Clustering on all cells complete", flush=True)
    printElapsed()

    # orthonormals = reg_trans.copy()
    #
    # # perform linear regression using the stimulus regressors
    # from sklearn.linear_model import LinearRegression
    # lr = LinearRegression()
    # r2_sensory_fit = np.empty(all_activity.shape[0], dtype=np.float32)
    # for i in range(all_activity.shape[0]):
    #     lr.fit(orthonormals, all_activity[i, :])
    #     r2_sensory_fit[i] = lr.score(orthonormals, all_activity[i, :])
    #
    # # recompute regressions unit-by-unit (should be done plane-by-plane for efficiency!!!) with motor regressors
    # # As null distribution compute a version with shuffled (wrong-plane) motor regressors
    # r2_sensory_motor_fit = np.zeros(all_activity.shape[0], dtype=np.float32)
    # for i in range(all_activity.shape[0]):
    #     mot_reg = all_motor[i, :]
    #     regs = np.c_[orthonormals, mot_reg]
    #     if np.any(np.isnan(regs)):
    #         continue
    #     else:
    #         lr.fit(regs, all_activity[i, :])
    #         r2_sensory_motor_fit[i] = lr.score(regs, all_activity[i, :])
    #
    # r2_sensory_motor_shuffle = np.zeros(all_activity.shape[0], dtype=np.float32)
    # for i in range(all_activity.shape[0]):
    #     shift = np.random.randint(50000, 100000)
    #     pick = (i + shift) % all_motor.shape[0]
    #     mot_reg = all_motor[pick, :]
    #     regs = np.c_[orthonormals, mot_reg]
    #     if np.any(np.isnan(regs)):
    #         continue
    #     else:
    #         lr.fit(regs, all_activity[i, :])
    #         r2_sensory_motor_shuffle[i] = lr.score(regs, all_activity[i, :])
    #
    # # compute confidence band based on shuffle
    # ci = 99.9
    # bedges = np.linspace(0, 1, 41)
    # bc = bedges[:-1] + np.diff(bedges)/2
    # conf = np.zeros(bc.size)
    # for i in range(bedges.size - 1):
    #     all_in = np.logical_and(r2_sensory_fit >= bedges[i], r2_sensory_fit < bedges[i + 1])
    #     if np.sum(all_in > 0):
    #         conf[i] = np.percentile(r2_sensory_motor_shuffle[all_in], ci)
    #     else:
    #         conf[i] = np.nan
    #
    # with sns.axes_style('whitegrid'):
    #     fig, ax = pl.subplots()
    #     ax.scatter(r2_sensory_fit, r2_sensory_motor_shuffle, alpha=0.4, color='r', s=3)
    #     ax.plot([0, 1], [0, 1], 'k--')
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #     ax.set_xlabel('$R^2$ Sensory regression')
    #     ax.set_ylabel('$R^2$ Sensory + Motor regression')
    #     ax.set_title('Shuffled motor regressor control')
    #
    # with sns.axes_style('whitegrid'):
    #     fig, ax = pl.subplots()
    #     ax.scatter(r2_sensory_fit, r2_sensory_motor_fit, alpha=0.4, s=3)
    #     ax.plot([0, 1], [0, 1], 'k--')
    #     # plot confidence band
    #     ax.fill_between(bc, bc, conf, color='orange', alpha=0.3)
    #     ax.plot(bc, conf, 'orange')
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #     ax.set_xlabel('$R^2$ Sensory regression')
    #     ax.set_ylabel('$R^2$ Sensory + Motor regression')
    #     ax.set_title('Boost of fit by including motor regressor')
    # print("Sensory and sensory plus motor fit completed", flush=True)
    # printElapsed()

    # compute movement triggered averages of motor correlated units as cross-correlations - both for all bouts
    # as well as by only taking bouts into account that are isolated, i.e. for which the frame distance to the next
    # bout is at least 5 frames (1s)
    def exp_bstarts(exp):
        tdd = mc_all.tdd
        traces = dict()
        bstarts = []
        # for binning, we want to have one more subdivision of the times and include the endpoint - later each time-bin
        # will correspond to one timepoint in interp_times above
        i_t = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1, endpoint=True)
        try:
            for gi in exp.graph_info:
                if gi[0] not in traces:
                    tdata = tdd[gi[0]]
                    conv_bstarts = tdata.starting
                    times = tdata.frameTime
                    digitized = np.digitize(times, i_t)
                    t = np.array([conv_bstarts[digitized == i].sum() for i in range(1, i_t.size)])
                    traces[gi[0]] = t
                bstarts.append(traces[gi[0]])
            raw_starts = np.vstack(bstarts)
        except KeyError:
            # this is necessary since some of the older experiments have been moved!
            return np.zeros((len(exp.graph_info), all_activity.shape[1]), dtype=np.float32)
        return raw_starts.astype(np.float32)
    # for each unit extract the original bout-trace binned to our 5Hz timebase
    raw_bstarts = np.vstack([exp_bstarts(e) for e in exp_data])
    raw_bstarts = raw_bstarts[no_nan_aa, :]
    max_lag = 25
    cc_all_starts = []
    cc_singular = []
    ac_all_starts = []
    ac_singular = []
    # identify the cluster number of the motor cluster
    for i in range(all_activity.shape[0]):
        if membership[no_nan_aa][i] >= n_regs and np.sum(raw_bstarts[i, :]) > 0:
            cc_all_starts.append(Crosscorrelation(all_activity[i, :], raw_bstarts[i, :], max_lag))
            ac_all_starts.append(Crosscorrelation(raw_bstarts[i, :], raw_bstarts[i, :], max_lag))
            sing_starts = raw_bstarts[i, :].copy()
            sing_starts[np.logical_not(np.logical_and(dist2next1(sing_starts) > 10, sing_starts < 2))] = 0
            if sing_starts.sum() == 0:
                continue
            cc_singular.append(Crosscorrelation(all_activity[i, :], sing_starts, max_lag))
            ac_singular.append(Crosscorrelation(sing_starts, sing_starts, max_lag))
    # plot cross and auto-correlations
    with sns.axes_style('whitegrid'):
        fig, (ax_ac, ax_cc) = pl.subplots(ncols=2)
        t = np.arange(-1*max_lag, max_lag+1) / 5
        sns.tsplot(data=ac_all_starts, time=t, ci=99.9, ax=ax_ac, color='b')
        sns.tsplot(data=ac_singular, time=t, ci=99.9, ax=ax_ac, color='r')
        ax_ac.set_xlabel('Time lag around bout [s]')
        ax_ac.set_ylabel('Auto-correlation')
        ax_ac.set_title('Bout start auto-correlation')
        sns.tsplot(data=cc_all_starts, time=t, ci=99.9, ax=ax_cc, color='b')
        sns.tsplot(data=cc_singular, time=t, ci=99.9, ax=ax_cc, color='r')
        ax_cc.set_xlabel('Time lag around bout [s]')
        ax_cc.set_ylabel('Cross-correlation')
        ax_cc.set_title('Bout start activity cross-correlation')
        fig.tight_layout()
    print("Computation of activity-bout cross-correlations complete", flush=True)
    printElapsed()

    # Plot per-fish cluster contributions
    m = int(np.ceil(np.sqrt(km.n_clusters)))
    fig, axes = pl.subplots(4, 3)
    axes = axes.ravel()
    for i in range(km.n_clusters):
        x = np.arange(len(exp_data))
        y = np.array([np.sum(exp_id[membership == i] == n) for n in x])
        sns.barplot(x, y, ax=axes[i])
        axes[i].set_title("Cluster " + str(i))
    fig.tight_layout()
