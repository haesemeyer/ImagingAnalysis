from mh_2P import UiGetFile, OpenStack, TailData, NucGraph, CorrelationControl, ReAlign, AvgNeighbhorCorrelations,\
    CorrelationGraph, ZScore_Stack

from scipy.ndimage import gaussian_filter

import cv2

import pickle

import numpy as np


def SplitBigGraphs(big_list, stack):
    """
    Takes a  list of anatomically segmented graphs that are above an area threshold and uses timeseries correlation to
    split these graphs into smaller constituents if possible. Returns the new list of split graphs
    """
    split_list = []

    # create rate stack for later correlation analysis
    rad = np.sqrt((min_area + max_area) / (2 * np.pi))
    rate_stack = gaussian_filter(stack, (frameRate, rad / 4, rad / 4))

    # loop over graphs
    for g in big_list:
        im_ncorr = AvgNeighbhorCorrelations(rate_stack[:, g.MinX:g.MaxX+1, g.MinY:g.MaxY+1])
        im_nc_full = np.zeros((stack.shape[1], stack.shape[2]), dtype=np.float32)
        # only pixels in teh original graph can be considered as seeds anyways
        for v in g.V:
            im_nc_full[v[0], v[1]] = im_ncorr[v[0] - g.MinX, v[1] - g.MinY]
        im_nc_full[np.isnan(im_nc_full)] = 0
        # we do not want to seed from pixels with neighborhodd correlation < 0.3 so skip graph if max correlation < 0.3
        if np.nanmax(im_nc_full) < 0.3:
            continue
        # print("Maximum neighborhood correlation = ", np.nanmax(im_nc_full), flush=True)

        # perform correlation based graph-search in the region allowing only inclusion of pixels present in the
        # original graph
        consider = lambda x, y: im_nc_full[x, y] > 0
        gls = CorrelationGraph.CorrelationConnComps(rate_stack, im_nc_full, 0.3, consider, seed_limit=0.3)[0]
        # assign raw timeseries to graphs
        for gg in gls:
            gg.RawTimeseries = np.zeros_like(gg.Timeseries)
            for v in gg.V:
                gg.RawTimeseries = gg.RawTimeseries + stack[:, v[0], v[1]]

        split_list += gls

    return split_list


if __name__ == "__main__":
    ca_time_constant = 1.796
    # serves as a guideline for filtering for segmentation - not true value for time-based paradigms with eye masking
    frameRate = 2.4

    zoom_level = float(input("Please enter the acquisition zoom:"))  # the zoom level used during acquisition
    min_area = 10 * (zoom_level**2)  # based on measurements, minimum area of a nucleus
    max_area = 30 * (zoom_level**2)  # based on measurements, maximum area of a nucleus

    # SEGMENTATION STRATEGY:
    # Run cell profiler with approximate size settings but DO NOT size filter identified objects
    # Use output from cell profiler nuclear identification as a starting point by segmenting the resulting 16-bit image
    # according to intensity levels (NucGraph.NuclearConnComp)
    # Subsequently, filter resulting graphs by size:
    # 1) For graphs that are larger than max_area, run correlation based sub-segmentation effectively splitting graph
    # 2) Discard all graphs (whether original or from splits) that fall below the min_area

    fnames = UiGetFile([('Nuclei file', 'nuclei.tif')], multiple=True)
    for fname in fnames:
        # find out if we tried to load the nuclei or original stack file
        if "nuclei" and "MAX" in fname:
            stackFile = fname[:-11] + '.tif'
            stackFile = stackFile.replace('MAX_', '')
            try:
                stack = np.load(stackFile[:-4] + "_stack.npy").astype(np.float32)
                print('Loaded aligned stack of slice ', flush=True)
            except FileNotFoundError:
                stack = OpenStack(stackFile)
                stack = ReAlign(stack, 4 * zoom_level)[0]
                np.save(stackFile[:-4] + "_stack.npy", stack.astype(np.uint8))
            segmentation_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)
            if np.unique(segmentation_image.ravel()).size == 1:
                # for some reason, if no region was found in cell profiler the background value of the image is not 0
                # so force it to be 0 throughout
                segmentation_image = np.zeros_like(segmentation_image)
            print("Maximum value in segmentation image: ", segmentation_image.max())
            # # load corresponding tail data
            # tfile = stackFile[:-6] + ".tail"
            # t_data = TailData.LoadTailData(tfile, ca_time_constant, 100)
        else:
            raise ValueError("Please load nuclei file")
        min_phot = stack.shape[0] * 0.02 / zoom_level
        sum_stack = np.sum(stack, 0)
        graph_list = NucGraph.NuclearConnComp(stack, segmentation_image, lambda x, y: sum_stack[x, y] >= min_phot)[0]

        # aggregate graphs that are at max area or below and graphs that need to be split
        norm_graphs = [g for g in graph_list if min_area <= g.NPixels <= max_area]
        big_graphs = [g for g in graph_list if g.NPixels > max_area]
        print("Identified ", len(big_graphs), " large graphs", flush=True)

        # split graphs that are too large
        big_graphs = SplitBigGraphs(big_graphs, stack)
        # remove graphs of less than min_area pixels after splitting
        big_graphs = [g for g in big_graphs if g.NPixels >= min_area]
        print("Obtained ", len(big_graphs), " split graphs", flush=True)
        # re-merge lists
        graph_list = norm_graphs + big_graphs

        # compute quality score deviation
        qualscore = CorrelationControl(stack, 36)[0]
        max_qual_deviation = (np.max(qualscore) - np.min(qualscore)) / np.mean(qualscore)
        print("Maximum quality score deviation = ", max_qual_deviation, flush=True)
        # assign necessary information to each graph
        for g in graph_list:
            g.SourceFile = stackFile  # store for convenience access
            g.StimFrequency = 0.1
            g.CaTimeConstant = ca_time_constant
            # g.PerFrameVigor = t_data.PerFrameVigor
            g.MaxQualScoreDeviation = max_qual_deviation
            # g.BoutStartTrace = t_data.FrameBoutStarts
            # The following attributes are incompatible with time-defined paradigms of arbitrary imaging framerate
            # g.FramesPre = 72
            # g.FramesStim = 144
            # g.FramesPost = 180
            # g.FramesFFTGap = 48
            # g.CellDiam = 8
            # g.FrameRate = frameRate

        f_graph = open(stackFile[:-3] + "graph", "wb")
        pickle.dump(graph_list, f_graph, protocol=pickle.HIGHEST_PROTOCOL)
        f_graph.close()
