from mh_2P import *

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

from sklearn.cluster import KMeans

import pickle




if __name__ == "__main__":
    graphFiles = UiGetFile([('PixelGraphs','.graph')],multiple=True)

    graph_list = []

    for gf in graphFiles:
        try:
            f = open(gf,'rb')
            graph_list += pickle.load(f)
        finally:
            f.close()

    #use comparisons btw. pre- and post-stimulus distributions to find background
    #distributions of magnitude at stimulus frequency as well as change in magnitude
    #at stimulus frequency
    stim_mag_bg = np.zeros(2*len(graph_list))
    stim_mag_stim = np.zeros(len(graph_list))
    stim_mag_ch_bg = np.zeros_like(stim_mag_bg)
    stim_mag_ch_stim = np.zeros_like(stim_mag_stim)

    for i,graph in enumerate(graph_list):
        try:
            pre = graph.FramesPre
            stim = graph.FramesStim
            post = graph.FramesPost
            fft_gap = graph.FramesFFTGap
            ix = np.argmin(np.absolute(graph.StimFrequency-graph.freqs_stim))
        except AttributeError:
            print("No paradigm information found in graph ", i,flush=True)
            pre = 144
            stim = 144
            post = 144
            fft_gap = 24
            ix = np.argmin(np.absolute(0.1-graph.freqs_stim))
        
        #re-compute fourier transforms on dF/F (mean(pre)=F) after 0.5s filtering
        F = np.mean(graph.RawTimeseries[:pre])
        dFF = (graph.RawTimeseries-F)/F
        dFF = gaussian_filter1d(dFF,1.2)
        graph.dFF = dFF
        sig_pre = dFF[:pre]
        sig_stm = dFF[pre:pre+stim]
        sig_pos = dFF[pre+stim:pre+stim+post]
        ms = lambda x: x-np.mean(x)
        fft_pre = np.fft.rfft(ms(sig_pre[fft_gap:]))
        fft_stm = np.fft.rfft(ms(sig_stm[fft_gap:]))
        fft_pos = np.fft.rfft(ms(sig_pos[fft_gap:]))
        assert(fft_pre.size == fft_stm.size and fft_stm.size == fft_pos.size and fft_pos.size == graph.fft_pre.size)
        graph.fft_pre = fft_pre
        graph.fft_stim = fft_stm
        graph.fft_post = fft_pos
        mags_pre = np.absolute(graph.fft_pre)
        mags_stim = np.absolute(graph.fft_stim)
        mags_post = np.absolute(graph.fft_post)

        stim_mag_bg[i] = mags_pre[ix]
        stim_mag_bg[i+len(graph_list)] = mags_post[ix]
        stim_mag_stim[i] = mags_stim[ix]

        stim_mag_ch_bg[i] = mags_pre[ix] - mags_post[ix]
        stim_mag_ch_bg[i+len(graph_list)] = mags_post[ix] - mags_pre[ix]
        stim_mag_ch_stim[i] = (mags_stim[ix] - mags_pre[ix] + mags_stim[ix] - mags_post[ix])/2



    #for each change category get indices of graphs that have a change larger
    #than the 95th percentile of the respective background distribution

   