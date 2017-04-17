from os import path

import numpy as np

import cv2

import nrrd

import pickle

from mh_2P import ReAlign, UiGetFile,  OpenStack, MakeNrrdHeader, CellGraph, CellHelper

from scipy.ndimage import gaussian_filter

from warnings import warn

if __name__ == "__main__":
    zoom_level = float(input("Please enter the acquisition zoom:"))  # the zoom level used during acquisition
    t_per_frame = float(input("Please enter the duration of each frame in seconds:"))

    frame_rate = 1 / t_per_frame

    ca_time_const = 0.4  # time-constant of cytoplasmic Gcamp6F
    # ca_time_const = 1.796  # time-constant of cytoplasmic Gcamp6F

    corr_thresh = 0.5

    # if set to true the aligned t-stack will be saved in nrrd format - note that this takes quite a long time
    write_aligned_nrrd = False

    fnames = UiGetFile(multiple=True)

    for i, f in enumerate(fnames):
        # try to load realigned stack if present, otherwise re-aling and save
        aligned_file = f[:-4] + "_stack.npy"
        if path.exists(aligned_file):
            print("Loaded pre-aligned stack", flush=True)
            stack = np.load(aligned_file, mmap_mode='r').astype(np.float32)
            resolution = 500/stack.shape[1]/zoom_level
        else:
            stack = OpenStack(f).astype(np.float32)
            resolution = 500 / stack.shape[1] / zoom_level
            stack, p_max_shift = ReAlign(stack, int(4 / resolution))[0:2]
            if p_max_shift > 2.5:
                warn("More than 2.5% of slices required max shift. Skipping stack {0}".format(f))
                continue
            np.save(aligned_file, stack.astype(np.uint8))
            if write_aligned_nrrd:
                out_stack = np.zeros((stack.shape[2], stack.shape[1], stack.shape[0]), dtype=np.uint8)
                for j, plane in enumerate(stack):
                    out_stack[:, :, j] = plane.T.astype(np.uint8)
                header = MakeNrrdHeader(out_stack.astype(np.uint8), resolution)
                ext_mark = f.find('.tif')
                out_name = f[:ext_mark] + '.nrrd'
                nrrd.write(out_name, out_stack, header)
            print("Realigned stack", flush=True)
        print("Resolution: ", resolution, " um/pixel", flush=True)
        cell_diam = 5 / resolution  # assume 5 um cell diameter
        # downsample stack in time-dimension before correlation based segmentation
        ds_by = 4
        fr_rate_ds = frame_rate / ds_by
        sum_stack = np.sum(stack, 0)
        stack = stack[::4, :, :].copy()
        # before finding cells filter stack along time-dimension
        rate_stack = gaussian_filter(stack, (fr_rate_ds, 0, 0))
        graph_list = CellGraph.CellConnComps(rate_stack, sum_stack, cell_diam, 30/(resolution**2), False)

        del rate_stack
        stack = np.load(f[:-4] + "_stack.npy", mmap_mode='r').astype(np.float32)
        for g in graph_list:
            g.SourceFile = f  # store for convenience access
            g.StimFrequency = np.nan
            g.CaTimeConstant = ca_time_const
            g.MaxQualScoreDeviation = np.nan
            # need to re-assign because of subsampling during segment.
            g.RawTimeseries = np.zeros(stack.shape[0], dtype=np.float32)
            for v in g.V:
                g.RawTimeseries += np.array(stack[:, v[0], v[1]].copy())

        # remove all cells, that aren't at least 2/3 in size of our maximal size
        graph_list = [g for g in graph_list if g.NPixels >= (30/(resolution**2)*2/3)]
        print('Identified ', len(graph_list), 'units in slice ', i, flush=True)
        f_graph = open(f[:-3] + "graph", "wb")
        pickle.dump(graph_list, f_graph, protocol=pickle.HIGHEST_PROTOCOL)
        f_graph.close()

        print(str(i + 1), " out of ", len(fnames), " completed", flush=True)
