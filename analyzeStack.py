# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:53:30 2016

@author: mhaesemeyer
"""

from mh_2P import *




def IsHeatActivated(trace,npre,nstim,npost):
    """
    Identify pixels that have been activated by the heat stimulus.
    Computes poisson rate during stim-off and based on that expected
    number of photons during stim period. Then uses poisson CDF to
    compute the likelihood of observing the number of actually
    observed photons during stim or more.
        trace: Time-series trace of a given pixel
        npre: Number of pre-stimulus frames
        nstim: Number of stimulus on frames
        npost: Number of post-stimulus frames
        RETURNS:
            [0]: The probability of the observed number of photons during stim-on
              given a poisson process with a rate based on the observed number of
              photons during stim-off (pre + post)
    """
    events_stim = np.sum(trace[npre:npre+nstim])
    events_noStim = np.sum(trace[:npre]) + np.sum(trace[npre+nstim:])
    lambda_noStim = events_noStim / (npre+npost) * nstim
    if events_stim > lambda_noStim and events_noStim > 0:
        distrib = poisson(lambda_noStim)
        p_pois = 1 - distrib.cdf(events_stim-1)
    else:
        p_pois = 1#technically incorrect unless events_stim = 0, but does what we need and saves some time
    return p_pois




if __name__ == "__main__":
    #paradigm constants
    pre_stim = 72#48
    stim = 144#48
    post_stim = 72#48
    stim_fft_gap = 24#0#number of initial frames not to use for fft determination
    min_phot = 10#minimum number of photons to observe in a pixels timeseries to not discard series
    des_freq = 0.3#1/3##the stimulus frequency
    corr_thresh = 0.2#correlation threshold for co-segmenting pixels

    filenames = UiGetFile([('Tiff Stack', '.tif;.tiff')],multiple=False)
    print(filenames)
    if type(filenames) is str:
        filenames = [filenames]
    for i,f in enumerate(filenames):
        #load stack
        stack = OpenStack(f).astype(float)
        #compute photon-rates using gaussian windowing
        rate_stack = gaussian_filter1d(stack,4.8,0)#standard deviation equal to 2s
        #we only want to consider time-series with at least min_phot photons
        sum_stack = np.sum(stack,0)
        consider = lambda x,y: sum_stack[x,y]>=min_phot
        #compute neighborhood correlations of pixel-timeseries for segmentation seeds
        im_ncorr = AvgNeighbhorCorrelations(rate_stack,2,consider)
        #display correlations and slice itself
        with sns.axes_style('white'):
            fig, (ax1, ax2) = pl.subplots(ncols=2)
            ax1.imshow(np.sum(stack,0),cmap='bone')
            ax1.set_title(f.split('/')[-1])
            ax2.imshow(im_ncorr,vmin=0,vmax=im_ncorr.max(),cmap="bone")
            fig.tight_layout()
        #extract correlation graphs - 4-connected
        graph, colors = CorrelationConnComps(rate_stack,im_ncorr,corr_thresh,False)
        #plot largest three components onto projection
        projection = np.zeros((im_ncorr.shape[0],im_ncorr.shape[1],3),dtype=float)
        projection[:,:,0] = projection[:,:,2] = sum_stack / sum_stack.max()
        g_sizes = [g.NPixels for g in graph]
        g_sizes = sorted(g_sizes)
        for g in graph:
            if g.NPixels > g_sizes[-4]:
                for v in g.V:
                    projection[v[0],v[1],1] = 0.3#color it green
        with sns.axes_style('white'):
            fig, ax = pl.subplots()
            ax.imshow(projection)

        #for each graph, create sum-trace which is only lightly filtered
        #assign to graph object and assign fft zscore at desired frequency to stack
        for g in graph:
            g.RawTimeseries = np.zeros_like(g.Timeseries)
            for v in g.V:
                g.RawTimeseries = g.RawTimeseries + stack[:,v[0],v[1]]
            #g.RawTimeseries = gaussian_filter1d(g.RawTimeseries,1)
            ms = lambda x: x-np.mean(x)#mean subtract
            fft = np.fft.rfft(ms(g.RawTimeseries[pre_stim+stim_fft_gap:pre_stim+stim]))
            try:
                freqs
            except NameError:
                freqs = np.linspace(0,1.2,fft.shape[0])
                dfs = np.absolute(des_freq-freqs)
                ix = np.argmin(dfs)
            mag = np.absolute(fft)
            zsc = (mag-np.mean(mag))/np.std(mag)
            g.FourierZ = zsc[ix]
            g.FourierMag = mag
            g.Frequencies = freqs

        #TODO: clean up graphs - potentially "fill holes" and remove very small connected components
        graph = [g for g in graph if g.NPixels>=10]#remove compoments with less than 10 pixels

        #plot histogram of all fourier z-scores for graphs with at least 10 pixels
        pl.figure()
        pl.hist([g.FourierZ for g in graph if g.NPixels>=10],20)

        ###########################################
        ##"OLD" FOURIER AND ACTIVITY BASED ANALYSIS##
        ###########################################
        #f_stack = FilterStackGaussian(stack,1)#FilterStack(stack.copy())
        ##threshold based on intensity - at least 5 photons over the whole time-series...
        #sum_stack = np.sum(f_stack,0)
        #for j in range(f_stack.shape[0]):
        #    im = f_stack[j,:,:]
        #    im[sum_stack<min_phot] = 0
        #    f_stack[j,:,:] = im
        #fft_stack = Per_PixelFourier(f_stack,pre_stim+stim_fft_gap,pre_stim+stim)[0]
        #freqs = np.linspace(0,1.2,fft_stack.shape[0])
        ##find the plane which most closely corresponds to our frequency
        #dfs = np.absolute(des_freq-freqs)
        #plane = np.argmin(dfs)
        #im_thresh = Threshold_Zsc(fft_stack,1.5,plane)
        #im_thresh[np.isnan(im_thresh)] = 0
        ##find pixels which showed activation during the heat-on period
        #im_activated = np.ones_like(im_thresh,dtype=float)
        #for x in range(im_thresh.shape[0]):
        #    for y in range(im_thresh.shape[1]):
        #        if sum_stack[x,y]>=min_phot:
        #            im_activated[x,y] = IsHeatActivated(stack[:,x,y],pre_stim,stim,post_stim)
        ##set pixels in im_thresh to 0 if im_activated>0.05. TODO: We should somehow allow region growth out of activated pixels into their neighbors
        #im_thresh[im_activated>0.2] = 0
        #with sns.axes_style('white'):
        #    fig, (ax1, ax2) = pl.subplots(ncols=2)
        #    ax1.imshow(np.sum(stack,0),cmap='bone')
        #    ax1.set_title(f.split('/')[-1])
        #    ax2.imshow(im_thresh,cmap="bone")
        #    fig.tight_layout()
        print((i+1)/len(filenames)*100,'% done',flush=True)