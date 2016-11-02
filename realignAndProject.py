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

    fnames = UiGetFile(multiple=True)

    for i, f in enumerate(fnames):
        contFolder = path.dirname(f)  # the containing folder
        stackName = path.split(f)[1]
        save_dir = contFolder + "/projections"
        if not path.exists(save_dir):
            makedirs(save_dir)
            print("Created projections directory", flush=True)
        stack = OpenStack(f).astype(float)
        # Realignment creates really bad artefacts in the 780nm RFP stacks whenever the eye is present - worse than
        # gcamp stacks
        stack = ReAlign(stack, 4 * zoom_level)[0]
        np.save(f[:-4] + "_stack.npy", stack.astype(np.uint8))
        print("Stack realigned", flush=True)

        if is_ref:
            if i == 0:
                z_stack = np.zeros((stack.shape[2], stack.shape[1], len(fnames)))
            # for reference stacks use common cutoff and save whole stack into nrrd file:
            projection = np.mean(stack, 0)
            projection /= 0.75
            projection[projection > 1] = 1
            projection *= (2**8-1)
            z_stack[:, :, i] = projection.T  # necessary otherwise x and y are switched
            if i == len(fnames)-1:
                header = MakeNrrdHeader(z_stack, 500/512/zoom_level)
                plane_mark = stackName.find('_Z_')
                out_name = stackName[:plane_mark] + '.nrrd'
                nrrd.write(save_dir + "/" + out_name, z_stack, header)
            cv2.imwrite(save_dir + "/" + "MAX_" + stackName, projection.astype(np.uint8))
        else:
            projection = np.sum(stack, 0)
            projection = cutOutliers(projection, 99.9) * (2**16-1)
            cv2.imwrite(save_dir + "/" + "MAX_" + stackName, projection.astype(np.uint16))