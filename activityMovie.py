# Create nrrd files of activity stacks of ON OFF cells for each time-point in SineLH experiments

import numpy as np
import nrrd
from mh_2P import MakeNrrdHeader, SLHRepeatExperiment
import h5py
from multiExpAnalysis import dff, get_stack_types
from typing import List
import pickle


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


if __name__ == "__main__":
    save_folder = "./HeatImaging/ActMovie/"
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    membership = np.array(dfile['membership'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    mship_nonan = membership[no_nan_aa]
    all_activity = np.array(dfile["all_activity"])
    tf_centroids = np.array(dfile['tf_centroids'])[no_nan_aa, :]
    pstream = np.array(dfile['exp_data_pickle'])
    exp_data = pickle.loads(pstream)  # type: List[SLHRepeatExperiment]
    del pstream
    dfile.close()
    stack_types = get_stack_types(exp_data)[no_nan_aa]

    dff_min = -1
    dff_max = 3

    active = np.logical_and(mship_nonan > -1, mship_nonan < 6)
    main_active = np.logical_and(active, stack_types == "MAIN")

    cells = dff(all_activity[main_active, :])
    cents = tf_centroids[main_active, :]
    # compute brighness values
    cells[cells < dff_min] = dff_min
    cells[cells > dff_max] = dff_max
    cells = (cells - dff_min) / (dff_max - dff_min)
    assert cells.min() >= 0
    assert cells.max() <= 1

    # loop over timepoints and create and save stack for each
    for i in range(cells.shape[1]):
        data, header = create_centroid_stack(cents, "MAIN", cells[:, i])
        nrrd.write(save_folder + "ON_OFF_Activity{0:03d}.nrrd".format(i), data, header)
        print("{0} completed".format(i))
