# Quick script to realign a bunch of stacks, calculate their projection, rescale that projection and save as images

from mh_2P import ReAlign, UiGetFile,  OpenStack, cutOutliers, MakeNrrdHeader

from os import path, makedirs

import numpy as np

import cv2

import nrrd

if __name__ == "__main__":

    zoom_level = float(input("Please enter the acquisition zoom:"))  # the zoom level used during acquisition

    is_ref = False
    normalization = input("[E]xperimental or [R]eference stack?")
    z_spacing = 2.5
    if normalization == "E":
        is_ref = False
    elif normalization == "R":
        is_ref = True
    else:
        print("Did not recognize input, defaulting to experimental stack", flush=True)
        is_ref = False

    co_align = input("Should channel 1 stacks be co-aligned with channel 0 stacks? [y/n]")
    if co_align == "y":
        co_align = True
    elif co_align == "n":
        co_align = False
    else:
        print("Did not recognize input, default to separate alignment", flush=True)
        co_align = False

    if is_ref and co_align:
        raise NotImplementedError("Co-alignment with reference stack generation currently not implemented")

    fnames = UiGetFile(multiple=True)

    for i, f in enumerate(fnames):
        if co_align and "_1.tif" in f:
            continue
        contFolder = path.dirname(f)  # the containing folder
        stackName = path.split(f)[1]
        save_dir = contFolder + "/projections"
        if not path.exists(save_dir):
            makedirs(save_dir)
            print("Created projections directory", flush=True)
        f_aligned = f[:-4] + "_stack.npy"
        stack_1 = None
        name_1 = None
        # if this run is for reference creation, we first check whether a realigned stack already exists
        if is_ref and path.exists(f_aligned):
            stack = np.load(f_aligned).astype(np.float32)
        else:
            stack = OpenStack(f).astype(np.float32)
            if co_align:
                name_1 = f[:-6]+"_1.tif"
                stack_1 = OpenStack(name_1)
                out = ReAlign(stack, 4 * zoom_level, stack_1)
                stack = out[0]
                stack_1 = out[-1]
            else:
                stack = ReAlign(stack, 4 * zoom_level)[0]
        if not is_ref:
            np.save(f_aligned, stack.astype(np.uint8))
        print("Stack realigned", flush=True)

        if is_ref:
            if i == 0:
                z_stack = np.zeros((stack.shape[2], stack.shape[1], len(fnames)), dtype=np.uint8)
            # for reference stacks use common cutoff and save whole stack into nrrd file:
            projection = np.mean(stack, 0)
            projection /= 0.75
            projection[projection > 1] = 1
            projection *= (2**8-1)
            z_stack[:, :, i] = projection.T.astype(np.uint8)  # necessary otherwise x and y are switched
            if i == len(fnames)-1:
                header = MakeNrrdHeader(z_stack, 500/512/zoom_level)
                plane_mark = stackName.find('_Z_')
                out_name = stackName[:plane_mark] + '.nrrd'
                nrrd.write(save_dir + "/" + out_name, z_stack, header)
        else:
            projection = np.sum(stack, 0)
            projection = cutOutliers(projection, 99.9) * (2**16-1)
            cv2.imwrite(save_dir + "/" + "MAX_" + stackName, projection.astype(np.uint16))
            if co_align:
                projection = np.sum(stack_1, 0)
                projection = cutOutliers(projection, 99.9) * (2**16-1)
                stack_1_Name = path.split(name_1)[1]
                cv2.imwrite(save_dir + "/" + "MAX_" + stack_1_Name, projection.astype(np.uint16))
