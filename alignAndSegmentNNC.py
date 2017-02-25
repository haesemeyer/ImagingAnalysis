# Script to realign a t-stack and perform segmentation based on nearest neighbor correlations

from os import path

import numpy as np

import cv2

import nrrd

import pickle

from mh_2P import ReAlign, UiGetFile,  OpenStack, TailData, MakeNrrdHeader, \
    ShuffleStackTemporal, AvgNeighbhorCorrelations, CorrelationGraph

from scipy.ndimage import gaussian_filter

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
            stack = ReAlign(stack, int(4 / resolution))[0]
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
        cell_diam = 6 / resolution  # assume 6 um cell diameter
        # downsample stack in time-dimension before correlation based segmentation
        ds_by = 4
        fr_rate_ds = frame_rate / ds_by
        stack = stack[::4, :, :].copy()

        # create shuffled stack to determine correlation seed cut-off
        st_shuff = ShuffleStackTemporal(stack)
        # we only want to consider time-series with at least min_phot photons
        min_phot = stack.shape[0] * 0.02
        sum_stack = np.sum(stack, 0)
        consider = lambda x, y: sum_stack[x, y] >= min_phot
        # compute photon-rates using gaussian windowing
        # along time standard deviation of 1s, 1/8 of cell diameter along spatial dimension
        rate_stack = gaussian_filter(stack, (fr_rate_ds, cell_diam / 8, cell_diam / 8))
        rs_shuff = gaussian_filter(st_shuff, (fr_rate_ds, cell_diam / 8, cell_diam / 8))
        # compute neighborhood correlations of pixel-timeseries for segmentation seeds
        im_ncorr = AvgNeighbhorCorrelations(rate_stack, 2, consider)
        im_nc_shuff = AvgNeighbhorCorrelations(rs_shuff, 2, consider)
        print('Maximum neighbor correlation in stack ', i, ' = ', im_ncorr.max(), flush=True)
        # determine correlation seed cutoff - find correlation value where correlations larger that value
        # are enriched at least ten times in the real dataset over the shuffled data-set
        seed_cutoff = 1
        for c in np.linspace(0, 1, 1001):
            if ((im_ncorr > c).sum() / (im_nc_shuff > c).sum()) >= 10:
                seed_cutoff = c
                break
        print('Correlation seed cutoff in stack ', i, ' = ', seed_cutoff, flush=True)
        # extract correlation graphs - 4-connected
        # cap our growth correlation threshold at the seed-cutoff, i.e. if corr_thresh
        # is larger than the significance threshold reduce it, when creating graph
        if corr_thresh <= seed_cutoff:
            ct_actual = corr_thresh
        else:
            ct_actual = seed_cutoff
        graph, colors = CorrelationGraph.CorrelationConnComps(rate_stack, im_ncorr, ct_actual, consider, False,
                                                              (0, rate_stack.shape[0]), seed_cutoff)
        # save segmentation image
        colors = colors.astype(np.uint16)
        ext_mark = f.find('.tif')
        out_name = f[:ext_mark] + '_segmentation.tif'
        cv2.imwrite(out_name, colors)
        min_size = np.pi*(cell_diam/2)**2 / 2  # half of a circle with the given average cell diameter
        graph_list = [g for g in graph if g.NPixels >= min_size]  # remove compoments smaller than half of cell diam
        # delete intermediates and reload non-subsampled stack
        del st_shuff
        del rs_shuff
        del rate_stack
        stack = np.load(f[:-4]+"_stack.npy", mmap_mode='r').astype(np.float32)
        print('Identified ', len(graph_list), 'units in slice ', i, flush=True)
        for g in graph_list:
            g.SourceFile = f  # store for convenience access
            g.StimFrequency = 0.1
            g.CaTimeConstant = ca_time_const
            g.MaxQualScoreDeviation = np.nan
            g.RawTimeseries = np.zeros_like(stack.shape[0])
            for v in g.V:
                g.RawTimeseries = g.RawTimeseries + stack[:, v[0], v[1]]

        f_graph = open(f[:-3] + "graph", "wb")
        pickle.dump(graph_list, f_graph, protocol=pickle.HIGHEST_PROTOCOL)
        f_graph.close()

        print(str(i+1), " out of ", len(fnames), " completed", flush=True)
