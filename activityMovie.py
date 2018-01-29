# Create nrrd files of activity stacks of ON OFF cells for each time-point in SineLH experiments

import numpy as np
import nrrd
from mh_2P import MakeNrrdHeader, SLHRepeatExperiment, UiGetFile, OpenStack
import h5py
from multiExpAnalysis import dff, get_stack_types
from typing import List
import pickle


def create_centroid_stack(centroids_um, stack_type="MAIN", brightness=1, radius=2):
    """
    Create a (refernce) stack with given centroids marked as single dots
    Args:
        centroids_um: The (x,y,z) centroid coordinates in um
        stack_type: Either "MAIN" or "TG" to obtain overall dimensions and resolution
        brightness: Either a scalar btw. 0 and 1 defining the brightness of all points or array with value per centroid
        radius: Radius of the ball drawn around each centroid in um

    Returns:
        [0]: Stack
        [1]: Compatible NRRD header
    """
    if not np.isscalar(brightness):
        if brightness.size != centroids_um.shape[0]:
            raise ValueError("Brightness either needs to be scalar or have one element per centroid")
    res_z = 2.5  # all our stacks have 2.5 um z-resolution
    if stack_type == "MAIN":
        res_xy = 500/512/1.5
        shape = (770, 1380, 100)
    elif stack_type == "TG":
        res_xy = 500/512/2
        shape = (512, 512, 30)
    else:
        raise ValueError("stack_type has to be one of 'MAIN' or 'TG'")
    # build ball offsets
    max_xy = int((radius + 1) // res_xy)
    max_z = int((radius + 1) // res_z)
    ball_offset_list = []
    for x in range(-max_xy, max_xy+1):
        for y in range(-max_xy, max_xy+1):
            for z in range(-max_z, max_z+1):
                d = np.sqrt((x*res_xy)**2 + (y*res_xy)**2 + (z*res_z)**2)
                if d <= radius:
                    ball_offset_list.append((x, y, z))
    stack = np.zeros(shape, dtype=np.uint8)
    header = MakeNrrdHeader(stack, res_xy, res_z)
    for i, cents in enumerate(centroids_um):
        if np.any(np.isnan(cents)):
            continue
        if np.isscalar(brightness):
            b = int(255 * brightness)
        else:
            b = int(255 * brightness[i])
        pixel_centroid = (int(cents[0] / res_xy), int(cents[1] / res_xy), int(cents[2] / res_z))
        stack[pixel_centroid[0], pixel_centroid[1], pixel_centroid[2]] = b
        for bo in ball_offset_list:
            stack[pixel_centroid[0]+bo[0], pixel_centroid[1]+bo[1], pixel_centroid[2]+bo[2]] = b
    return stack, header


def draw_temp_inset():
    def box_coords(t):
        y_bottom = 100
        x_left = 50
        x_right = 75
        y_size = int((t - 22) / 7 * 75)
        y_top = y_bottom - y_size
        return x_left, x_right, y_top, y_bottom

    print("Load movie stack", flush=True)
    sfile = UiGetFile()
    stk_shape = OpenStack(sfile).shape
    inset_stack = np.zeros((stk_shape[2], stk_shape[1], stk_shape[0]), dtype=np.uint8)
    header = MakeNrrdHeader(inset_stack, 1)
    tmp_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    temperature = np.array(tmp_file["sine_L_H_temp"])
    tmp_file.close()

    for i in range(stk_shape[0]):
        xl, xr, yt, yb = box_coords(temperature[i*4])
        inset_stack[xl:xr+1, yt:yb+1, i] = 255
    nrrd.write("./HeatImaging/TInset/" + "temp_inset.nrrd", inset_stack, header)


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
    on = np.logical_and(mship_nonan > -1, mship_nonan < 4)
    off = np.logical_and(mship_nonan > 3, mship_nonan < 6)
    # motor = mship_nonan > 5
    main_on = np.logical_and(on, stack_types == "MAIN")
    main_off = np.logical_and(off, stack_types == "MAIN")
    # main_motor = np.logical_and(motor, stack_types == "MAIN")
    main_active = np.logical_and(active, stack_types == "MAIN")

    cells = dff(all_activity[main_active, :])
    # select 2 ON and 2 OFF example cells to show in brain as well as activity insets
    rh_6_l_on = 15048
    rh_6_l_off = 15104
    hab_r_on = 254
    hab_r_off = 5373
    cents = tf_centroids[main_active, :]
    # save one stack with example ON cells and one stack with example OFF cells
    data, header = create_centroid_stack(cents[[rh_6_l_on, hab_r_on], :], radius=4)
    nrrd.write(save_folder + "ON_Cells.nrrd", data, header)
    data, header = create_centroid_stack(cents[[rh_6_l_off, hab_r_off], :], radius=4)
    nrrd.write(save_folder + "OFF_Cells.nrrd", data, header)

    # compute brighness values
    cells[cells < dff_min] = dff_min
    cells[cells > dff_max] = dff_max
    cells = (cells - dff_min) / (dff_max - dff_min)
    assert cells.min() >= 0
    assert cells.max() <= 1

    # # save one stack each with all ON and all OFF cells
    # data, header = create_centroid_stack(tf_centroids[main_on, :][np.random.rand(main_on.sum()) < 0.2, :])
    # nrrd.write(save_folder + "ON_Cells.nrrd", data, header)
    # data, header = create_centroid_stack(tf_centroids[main_off, :][np.random.rand(main_off.sum()) < 0.2, :])
    # nrrd.write(save_folder + "OFF_Cells.nrrd", data, header)
    # data, header = create_centroid_stack(tf_centroids[main_motor, :][np.random.rand(main_motor.sum()) < 0.2, :])
    # nrrd.write(save_folder + "Motor_Cells.nrrd", data, header)

    # loop over timepoints and create and save stack for each
    for i in range(cells.shape[1]):
        data, header = create_centroid_stack(cents, "MAIN", cells[:, i])
        nrrd.write(save_folder + "ON_OFF_Activity{0:03d}.nrrd".format(i), data, header)
        print("{0} / {1} completed".format(i+1, cells.shape[1]))
