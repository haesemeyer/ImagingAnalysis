from mh_2P import *

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

from sklearn.cluster import KMeans

import pickle


from analyzeStack import AssignFourier

import sys
sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')

import mhba_basic as mb


def ShuffleGraph(graph):
    g_shuff = CorrelationGraph(-1,np.roll(graph.RawTimeseries,np.random.randint(graph.FramesPre//2,graph.FramesPre+graph.FramesStim+graph.FramesPost//2)))
    #copy necessary values
    g_shuff.RawTimeseries = g_shuff.Timeseries
    g_shuff.FramesPre = graph.FramesPre
    g_shuff.FramesStim = graph.FramesStim
    g_shuff.FramesFFTGap = graph.FramesFFTGap
    g_shuff.FramesPost = graph.FramesPost
    g_shuff.StimFrequency = graph.StimFrequency
    g_shuff.freqs_pre = graph.freqs_pre
    g_shuff.freqs_stim = graph.freqs_stim
    g_shuff.freqs_post = graph.freqs_post
    #re-compute fourier transforms on shuffled time-series
    pre,stim,post = g_shuff.FramesPre,g_shuff.FramesStim,g_shuff.FramesPost
    gap = g_shuff.FramesFFTGap
    AssignFourier(g_shuff,gap,pre,des_freq,frame_rate,'pre',True)
    AssignFourier(g_shuff,pre+gap,pre+stim,des_freq,frame_rate,'stim',True)
    AssignFourier(g_shuff,pre+stim+gap,pre+stim+post,des_freq,frame_rate,'post',True)
    return g_shuff



def ComputeDFF(graph):
    timeseries = graph.RawTimeseries
    base_start = 0
    base_end = graph.FramesPre
    F = np.mean(timeseries[base_start:base_end])
    return (timeseries-F)/F

def ComputeFourierChangeRatio(graph):
    f = graph.freqs_pre
    #find index of stimulus frequency
    ix = np.argmin(np.abs(f-graph.StimFrequency))
    mag_stim_atFreq = np.absolute(graph.fft_stim)[ix]
    mag_stim_other = np.sum(np.absolute(graph.fft_stim)[1:])#exclude 0-point, i.e. stimulus mean
    ratio_stim = mag_stim_atFreq/mag_stim_other
    mag_bg_atFreq = np.absolute(graph.fft_pre)[ix] + np.absolute(graph.fft_post)[ix]
    mag_bg_other = np.sum(np.absolute(graph.fft_pre[1:])) + np.sum(np.absolute(graph.fft_post[1:]))
    ratio_bg = mag_bg_atFreq / mag_bg_other
    return ratio_stim / ratio_bg, ratio_stim

def ComputeAnglesAtStim(graph):
    f = graph.freqs_pre
    #find index of stimulus frequency
    ix = np.argmin(np.abs(f-graph.StimFrequency))
    return np.angle(graph.fft_pre)[ix], np.angle(graph.fft_stim)[ix], np.angle(graph.fft_post)[ix]


def ComputeStimActivityIncrease(graph):
    pre,stim,post = graph.FramesPre,graph.FramesStim,graph.FramesPost
    avg_pre = np.mean(graph.RawTimeseries[:pre])
    avg_stim = np.mean(graph.RawTimeseries[pre:pre+stim])
    avg_post = np.mean(graph.RawTimeseries[pre+stim:pre+stim+post])
    return (avg_stim/avg_pre + avg_stim/avg_post)/2

def GetFltAverages(graph):
    dff = ComputeDFF(graph)
    dff = gaussian_filter1d(dff,1.2)
    pre,stim,post = graph.FramesPre,graph.FramesStim,graph.FramesPost
    if pre!=stim or pre!=post or pre!=144:
        raise NotImplementedError("Currently only equal phase-lengths of 144 frames are supported")
    pre_trace = dff[:pre].reshape((6,144//6))
    stim_trace = dff[pre:pre+stim].reshape((6,144//6))
    post_trace = dff[pre+stim:pre+stim+post].reshape((6,144//6))
    return pre_trace, stim_trace, post_trace

def PlotFltAverages(graph,ax=None):
    pre_trace, stim_trace, post_trace = GetFltAverages(graph)
    time = np.linspace(0,10,pre_trace.shape[1],endpoint=False)
    if ax is None:
        pl.figure()
        pl.plot(time-10,np.mean(pre_trace,0))
        pl.plot(time+10,np.mean(post_trace,0))
        pl.plot(time,np.mean(stim_trace,0))
        sns.despine()
    else:
        ax.plot(time-10,np.mean(pre_trace,0))
        ax.plot(time+10,np.mean(post_trace,0))
        ax.plot(time,np.mean(stim_trace,0))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('dF/F trial average')
        sns.despine(ax=ax)

def PlotFFTMags(graph,ax=None):
    mag_pre = np.absolute(graph.fft_pre)
    mag_stim = np.absolute(graph.fft_stim)
    mag_post = np.absolute(graph.fft_post)
    freqs = graph.freqs_pre
    if ax is None:
        pl.figure()
        pl.plot(freqs,mag_pre)
        pl.plot(freqs,mag_post)
        pl.plot(freqs,mag_stim)
        pl.xlabel('Frequency [Hz]')
        pl.ylabel('Magnitude')
        sns.despine()
    else:
        ax.plot(freqs[1:],mag_pre[1:])
        ax.plot(freqs[1:],mag_post[1:])
        ax.plot(freqs[1:],mag_stim[1:])
        ylm = ax.get_ylim()[1]
        ax.plot([graph.StimFrequency,graph.StimFrequency],[0,ylm],'k--',alpha=0.6)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude')
        sns.despine(ax=ax)


def PlotDFF(graph,ax=None):
    dff = gaussian_filter1d(ComputeDFF(graph),1.2)
    time = np.linspace(0,180,144*3,endpoint=False)
    if ax is None:
        pl.figure()
        pl.plot(time[:145],dff[:145])
        pl.plot(time[287:144*3],dff[287:144*3])
        pl.plot(time[144:288],dff[144:288])
        pl.ylabel('dF/F')
        pl.xlabel('Time [s]')
        sns.despine()
    else:
        ax.plot(time[:145],dff[:145])
        ax.plot(time[287:144*3],dff[287:144*3])
        ax.plot(time[144:288],dff[144:288])
        ax.set_ylabel('dF/F')
        ax.set_xlabel('Time [s]')
        sns.despine(ax=ax)

def PlotROI(graph,ax=None):
    """
    Plots graph location on slice sum of original stack
    """
    stack_file = graph.SourceFile[:-4]+"_stack.npy"
    stack = np.load(stack_file).astype(float)
    sum_stack = np.sum(stack,0)
    projection = np.zeros((sum_stack.shape[0],sum_stack.shape[1],3))
    projection[:,:,0] = projection[:,:,1] = projection[:,:,2] = sum_stack/sum_stack.max()*2
    projection[projection>0.8] = 0.8
    for v in graph.V:
        projection[v[0],v[1],0] = 1
    if ax is None:
        with sns.axes_style("white"):
            fig,ax  = pl.subplots()
            ax.imshow(projection)
            sns.despine(fig,ax,True,True,True,True)
    else:
        ax.imshow(projection)
        sns.despine(None,ax,True,True,True,True)


def PlotMajorInfo(graph):
    with sns.axes_style('white'):
        fig, axes = pl.subplots(ncols=4)
        fig.set_size_inches(15,5,5)
        PlotROI(graph,axes[0])
        PlotDFF(graph,axes[1])
        PlotFFTMags(graph,axes[2])
        PlotFltAverages(graph,axes[3])
        fig.tight_layout()


    

def LoadGraphTailData(graph):
    name = graph.SourceFile[:-6]+".tail"
    return TailData.LoadTailData(name,graph.CaTimeConstant)



if __name__ == "__main__":
    graphFiles = UiGetFile([('PixelGraphs','.graph')],multiple=True)

    #gcamp6f
    #qual_cut = 0.1
    #gcamp6s (TB confirmed)
    qual_cut = 0.2

    graph_list = []

    for gf in graphFiles:
        try:
            f = open(gf,'rb')
            graph_list += pickle.load(f)
        finally:
            f.close()

    #compute changes in fourier ratio and overall activity between stimulus
    #and pre-post periods - both on each original graph and a time-series
    #shuffled version of it to create a background distribution
    stim_act_inc = np.zeros(len(graph_list))#increase in mean activity during stimulus period compared to pre-post
    stim_act_inc_sh = np.zeros_like(stim_act_inc)
    stim_frat = np.zeros_like(stim_act_inc)#fourier ratio at stimulus freqency during stimulus
    stim_frat_sh = np.zeros_like(stim_act_inc)
    stim_frat_inc = np.zeros_like(stim_act_inc)#change in fourier ratio at stimulus frequency during stimulus compared to non-stimulus
    stim_frat_inc_sh = np.zeros_like(stim_frat_inc)

    motor_corr = np.zeros_like(stim_frat_inc)#the correlation of the graph's time-series to motor events
    motor_corr_sh = np.zeros_like(motor_corr)

    all_quality_scores = np.zeros(len(graph_list))#for each ROI the stacks movement quality score: 0 is best, <0.1 likely tolerable, >=0.1 problematic

    pre_post_mean_change = np.zeros_like(all_quality_scores)#tracks the fold-change of post stimulus mean compared to pre-stimulus mean

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
        
        all_quality_scores[i] = graph.MaxQualScoreDeviation
        g_shuffled = ShuffleGraph(graph)
        stim_act_inc[i] = ComputeStimActivityIncrease(graph)
        stim_act_inc_sh[i] = ComputeStimActivityIncrease(g_shuffled)
        stim_frat_inc[i],stim_frat[i] = ComputeFourierChangeRatio(graph)
        stim_frat_inc_sh[i],stim_frat_sh[i] = ComputeFourierChangeRatio(g_shuffled)
        pre_mean = np.mean(graph.RawTimeseries[:pre])
        post_mean = np.mean(graph.RawTimeseries[pre+stim:pre+stim+post])
        pre_post_mean_change[i] = post_mean/pre_mean
        if graph.BoutStartTrace.size == graph.RawTimeseries.size:
            motor_corr[i] = np.corrcoef(graph.BoutStartTrace,graph.RawTimeseries)[0,1]
            motor_corr_sh[i] = np.corrcoef(graph.BoutStartTrace,g_shuffled.RawTimeseries)[0,1]
        else:#left-over frame not removed
            motor_corr[i] = np.corrcoef(graph.BoutStartTrace[:-1],graph.RawTimeseries)[0,1]
            motor_corr_sh[i] = np.corrcoef(graph.BoutStartTrace[:-1],g_shuffled.RawTimeseries)[0,1]
        



    #for each change category get indices of graphs that have a change larger
    #than the 95th percentile of the respective background distribution
    cut_frat_inc = np.percentile(stim_frat_inc_sh[all_quality_scores<qual_cut],95)
    cut_frat = np.percentile(stim_frat_sh[all_quality_scores<qual_cut],95)
    take = [g for i,g in enumerate(graph_list) if stim_frat_inc[i]>cut_frat_inc and stim_frat[i]>cut_frat and all_quality_scores[i]<qual_cut]

    cut_inc = np.percentile(stim_act_inc_sh[all_quality_scores<qual_cut],95)
    take_increase = [g for i,g in enumerate(graph_list) if stim_act_inc[i]>cut_inc and all_quality_scores[i]<qual_cut]#ignore for now

    #identify potential motor-correlated units
    cut_mc = np.percentile(motor_corr_sh[all_quality_scores<qual_cut],99)
    take_mc = [g for i,g in enumerate(graph_list) if motor_corr[i]>cut_mc and all_quality_scores[i]<qual_cut]

    #get set of all graphs that are in either take_increase and/or take_mc
    take_all = set(take + take_mc)

    #for clustering create a matrix with take_all.size rows. Each row contains the concatenatio
    #of the per-period average stimulus trace (10s, 24 frames) as well as the bout triggered
    #average response (4 pre and 4 post frames, total of 9 frames, 3.75 seonds)
    #zscore each segment but give bout triggered average a standard-deviation of 2 to compensate
    #for having less frames in this segment...

    #create dictionary with tail data for each relevant slice
    taildata = dict()
    for g in take_all:
        if g.SourceFile in taildata:
            continue;
        else:
            taildata[g.SourceFile] = LoadGraphTailData(g)


    #separate motor-correlated units out (?) - motor likely correlated to stimulus - so how to separate properly??

    #cluster potential sensory driven units(?) / separate by phase (?)

   