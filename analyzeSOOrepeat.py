#file to analyze 2-photon imaging data from sine-on-off experiments with repeat presentation
#unit graphs should have been constructed and saved before using analyzeStack.py script


from mh_2P import *

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

from sklearn.cluster import KMeans

import pickle

import warnings

import sys
sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')

import mhba_basic as mb

from mh_lasTemp import LaserTempData



def LoadGraphTailData(graph):
    name = graph.SourceFile[:-6]+".tail"
    return TailData.LoadTailData(name,graph.CaTimeConstant)

def ZScore(trace):
    return (trace-np.mean(trace))/np.std(trace)

def ComputeTraceFourierFraction(trace,startFrame,endFrame,des_freq,frame_rate,aggregate=True):
    filtered = gaussian_filter1d(trace,frame_rate/8)
    filtered = filtered[startFrame:endFrame]
    if aggregate:
        #Test if we can aggregate: Find the period length pl in frames. If the length of filtered
        #is a multiple of 2 period lengths (size = 2* N * pl), reshape and average across first
        #and second half to reduce noise in transform (while at the same time reducing resolution)
        pl = round(1 / des_freq * frame_rate)
        if (filtered.size/pl) % 2 == 0:
            filtered = np.mean(filtered.reshape((2,filtered.size//2)),0)
        else:
            warning.warn('Could not aggregate for fourier due to phase alignment mismatch')
    fft = np.fft.rfft(ZScore(filtered))
    mag = np.absolute(fft)
    freqs = np.linspace(0,frame_rate/2,fft.shape[0])
    ix = np.argmin(np.absolute(des_freq-freqs))
    return np.absolute(fft)[ix] / np.sum(np.absolute(fft))

def ComputeFourierAvgStim(graph,startFrame,endFrame,des_freq,frame_rate,aggregate=True):
    """
        Computes the fourier transform of the stimulus period on the repeat-averaged
        timeseries.
    """
    #anti-aliasing
    filtered = gaussian_filter1d(graph.AveragedTimeseries,frame_rate/8)
    filtered = filtered[startFrame:endFrame]
    #TODO: Somehow make the following noise reduction more generally applicable...
    #if the length of filtered is divisble by 2, break into two blocks and average for noise reduction
    if aggregate:
        #Test if we can aggregate: Find the period length pl in frames. If the length of filtered
        #is a multiple of 2 period lengths (size = 2* N * pl), reshape and average across first
        #and second half to reduce noise in transform (while at the same time reducing resolution)
        pl = round(1 / des_freq * frame_rate)
        if (filtered.size/pl) % 2 == 0:
            filtered = np.mean(filtered.reshape((2,filtered.size//2)),0)
        else:
            warning.warn('Could not aggregate for fourier due to phase alignment mismatch')
    fft = np.fft.rfft(ZScore(filtered))
    freqs = np.linspace(0,frame_rate/2,fft.shape[0])
    mag = np.absolute(fft)
    ix = np.argmin(np.absolute(des_freq-freqs))#index of bin which contains our desired frequency
    graph.stimFFT = fft
    graph.stimFFT_freqs = freqs
    graph.mag_atStim = np.absolute(fft)[ix]
    graph.mfrac_atStim = graph.mag_atStim / np.sum(np.absolute(fft))
    graph.ang_atStim = np.angle(fft)[ix]

def ComputeAveragedTimeseries(timeseries):
    l = timeseries.size
    return np.sum(np.reshape(timeseries[0:l-n_hangoverFrames],(2,(l-1)//2)),0)#sum/mean equivalent here (anyways determined by graph size)


#def ComputeStimInducedNoise(graph,avg_timeseries):
#    """
#    Using information about pre, stim and post frames
#    from the graph computes the ratio of standard
#    deviations of timeseries between pre frames
#    and stimulus-presentation frames (stim+post)
#    """
#    s_pre = np.std(avg_timeseries[:graph.FramesPre])
#    s_stim = np.std(avg_timeseries[graph.FramesPre:])
#    return s_stim / s_pre

def ComputeStimulusEffect(graph,timeseries):
    """
    Tries to address whether the activity of a unit
    gets mostly influenced by the stimulus. To that end
    compares the variation in the two pre-stimulus periods
    with the variation between stimulus periods and their
    given pre periods
    """
    l = timeseries.size
    reps = np.reshape(timeseries[0:l-n_hangoverFrames],(2,(l-1)//2))
    m_pre = np.mean(reps[:,:graph.FramesPre],1)
    m_stim = np.mean(reps[:,graph.FramesPre:],1)
    d_pre = np.abs(m_pre[0]-m_pre[1])
    d_ps = np.mean(np.abs(m_pre-m_stim))
    return d_ps / d_pre

def ComputeSwimTriggeredAverage(graph_list,tailData,f_pre,f_post):
    mta = []
    for g in graph_list:
        td = tailData[g.SourceFile]
        if td.boutFrames is None or td.boutFrames.size == 0:
            continue
        im_len = g.RawTimeseries.size
        gts = (g.RawTimeseries-np.percentile(g.RawTimeseries,20))/np.percentile(g.RawTimeseries,20)
        if np.any(np.isnan(gts)) or np.any(np.isinf(gts)):
            continue
        ix = mb.IndexingMatrix(td.boutFrames,f_pre,f_post,im_len)[0]
        btrig = gts[ix]
        mta.append(np.nanmean(btrig,0))
    return np.vstack(mta)

from scipy.signal import lfilter

def CorrelateResWithMTA(graph,mta):
    rs = graph.RawTimeseries
    rs = (rs-np.mean(rs))/np.std(rs)
    mta = (mta-np.mean(mta))/np.std(mta)
    return lfilter(mta[::-1],1,rs)

def ComputeDFF(avg_timeseries):
    f0 = np.mean(avg_timeseries[:72])
    return (avg_timeseries-f0)/f0

def CaConvolve(trace,ca_timeconstant,frame_rate):
    kernel = TailData.CaKernel(ca_timeconstant,frame_rate)
    return np.convolve(trace,kernel)[:trace.size]


def PlotROI(graph, ax=None, motor=False):
    """
    Plots graph location on slice sum of original stack
    """
    stack_file = graph.SourceFile[:-4]+"_stack.npy"
    try:
        stack = np.load(stack_file).astype(np.float32)
    except FileNotFoundError:
        stack = OpenStack(graph.SourceFile)
    sum_stack = np.sum(stack, 0)
    projection = np.zeros((sum_stack.shape[0], sum_stack.shape[1], 3))
    projection[:, :, 0] = projection[:, :, 1] = projection[:, :, 2] = sum_stack/sum_stack.max()*2
    projection[projection > 0.8] = 0.8
    if motor:
        for v in graph.V:
            projection[v[0], v[1], 2] = 1
    else:
        for v in graph.V:
            projection[v[0], v[1], 0] = 1
    if ax is None:
        with sns.axes_style("white"):
            fig, ax = pl.subplots()
            ax.imshow(projection)
            sns.despine(fig, ax, True, True, True, True)
    else:
        ax.imshow(projection)
        sns.despine(None, ax, True, True, True, True)


def PlotAvgDff(graph,ax=None):
    if ax is None:
        fig,ax = pl.subplots()
    dff = ComputeDFF(graph.AveragedTimeseries)
    time = np.arange(dff.size)/graph.FrameRate
    ax.plot(time,dff,'r',label='Response')
    ax.plot(time,graph.StimOn*dff.max(),label='Stimulus')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('DFF')
    ax.set_title('Repeat average activity')
    ax.legend()

def PlotMotorRelation(graph,ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    #zs_act = ZScore(graph.RawTimeseries)
    zs_act = (graph.RawTimeseries - np.percentile(graph.RawTimeseries,20)) / np.percentile(graph.RawTimeseries,20)
    zs_mot = ZScore(graph.PerFrameVigor)
    zs_mot = zs_mot / zs_mot.max() * zs_act.max()
    time = np.arange(zs_act.size)/graph.FrameRate
    ax.plot(time,zs_act,'r',label='Response')
    ax.plot(time,zs_mot,label='Swim vigor')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Act: DFF / Mot: AU')
    ax.set_title('Raw activity vs. motor output')
    ax.legend()

def PlotMetrics(graph,ax=None,index=None):
    if ax is None:
        fig, ax = pl.subplots()
    ax.text(0.1,0.8,str.format('Stimulus induction {0:.2f}',graph.StimIndFluct),transform=ax.transAxes)
    ax.text(0.1,0.7,str.format('Motor correlation {0:.2f}',graph.MotorCorrelation),transform=ax.transAxes)
    ax.text(0.1,0.6,str.format('ON Correlation {0:.2f}',graph.CorrOn),transform=ax.transAxes)
    ax.text(0.1,0.5,str.format('OFF Correlation {0:.2f}',graph.CorrOff),transform=ax.transAxes)
    ax.text(0.1,0.4,str.format('Magnitude fraction {0:.2f}',graph.mfrac_atStim),transform=ax.transAxes)
    #shuffle cutoffs
    ax.text(0.6,0.8,str.format('{0:.2f}',graph.sh_m_StIndFluct+2*graph.sh_std_StIndFluct),transform=ax.transAxes,color='r')
    ax.text(0.6,0.6,str.format('{0:.2f}',graph.sh_m_CorrOn+2*graph.sh_std_CorrOn),transform=ax.transAxes,color='r')
    ax.text(0.6,0.5,str.format('{0:.2f}',graph.sh_m_CorrOff+2*graph.sh_std_CorrOff),transform=ax.transAxes,color='r')
    ax.text(0.6,0.4,str.format('{0:.2f}',graph.sh_m_mfrac+2*graph.sh_std_mfrac),transform=ax.transAxes,color='r')

    if not index is None:
        ax.text(0.1,0.3,str.format('Array index {0}',index),transform=ax.transAxes)
    ax.text(0.1,0.2,str.format('Sourcefile {0}',graph.SourceFile.split('/')[-1]),transform=ax.transAxes)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def PlotGraphInfo(graph,index=None):
    with sns.axes_style('white'):
        fig, axes = pl.subplots(2,2,figsize=(15,10))
        PlotROI(graph,axes[0,0])
        PlotAvgDff(graph,axes[0,1])
        PlotMotorRelation(graph,axes[1,0])
        PlotMetrics(graph,axes[1,1],index)
        sns.despine(fig)
        fig.tight_layout()

def PlotCycleGraphList(glist):
    from matplotlib.pyplot import pause
    skip_dic = dict()
    for i,g in enumerate(glist):
        if g.SourceFile in skip_dic:
            continue
        PlotGraphInfo(g,i)
        pause(0.5)
        inp = ""
        while inp!="s" and inp!="p" and inp!="c":
            inp = input("[c]ontinue/[s]kip plane/sto[p]:")
        if inp=="p":
            break
        elif inp=="s":
            skip_dic[g.SourceFile] = 1
        pl.close('all')
        print(i/len(glist),flush=True)

def restFreq(tail_d):
    if tail_d.bouts is None:
        return 0
    st_b = np.sum(phase[tail_d.boutFrames.astype(int)]==0)
    return st_b/150


def stimFreq(tail_d):
    if tail_d.bouts is None:
        return 0
    st_b = np.sum(phase[tail_d.boutFrames.astype(int)]==1)
    return st_b/180


if __name__ == "__main__":
    n_repeats = 2#the number of repeats performed in each plane
    n_hangoverFrames = 1#the number of "extra" recorded frames at experiment end
    #our quality score is actually problematic, especially on non-whole-brain data:
    #1) In focused regions such as the trigeminal, large activity spikes cause poor scores even in the absence of movement (manual check)
    #2) If the eyes are contained in the stack, as they can move within the agarose they can corrupt the score
    #=> Until better metric is found set to large value...
    qual_cut = 0.1#maximum quality score deviation to still consider graph valid
    n_shuffles = 200

    #load unit activity from each imaging plane, compute motor correlations
    #and repeat averaged time-trace <- note that motor correlations have to
    #be calculated on non-time averaged data
    graphFiles = UiGetFile([('PixelGraphs','.graph')],multiple=True)
    graph_list = []

    #create vector describing stimulus phase of experiment. TODO: Define same way as stimOn below, dependent on graph
    phase = np.zeros((72+144+180)*2+1)
    phase[72:72+144] = 1
    phase[72+144+36:72+144+72] = 1
    phase[288+36:288+72] = 1
    phase[72+144+180:-1] = phase[:72+144+180]

    for gf in graphFiles:
        try:
            f = open(gf,'rb')
            graphs = pickle.load(f)
            if len(graphs)>0:# and graphs[0].MaxQualScoreDeviation < qual_cut:#don't even process or add graphs from a plane that has a bad quality score
                for g in graphs:
                    g.MotorCorrelation = np.corrcoef(g.RawTimeseries,g.PerFrameVigor)[0,1]
                    g.AveragedTimeseries = ComputeAveragedTimeseries(g.RawTimeseries)
                    #create stimulus regressors and assign to graph
                    try:
                        g.StimOn = stimOn
                        g.StimOff = stimOff
                    except NameError:
                        post_start = g.FramesPre+g.FramesStim
                        step_len = 15 * g.FrameRate
                        stimOn = stimOnset = stimOffset = np.zeros_like(g.AveragedTimeseries)
                        stimOn[g.FramesPre:post_start] = 1
                        stimOn[post_start+step_len:post_start+2*step_len] = 1
                        stimOn[post_start+3*step_len:post_start+4*step_len] = 1
                        stimOff = 1-stimOn
                        stimOn = LaserTempData.PredictTemperature(3.625,0.53,stimOn,1/2.4)
                        stimOff = LaserTempData.PredictTemperature(3.625,0.53,stimOff,1/2.4)
                        stimOn = CaConvolve(stimOn,g.CaTimeConstant,g.FrameRate)
                        stimOn = stimOn / stimOn.max()
                        stimOff = CaConvolve(stimOff,g.CaTimeConstant,g.FrameRate)
                        stimOff = stimOff / stimOff.max()
                        g.StimOn = stimOn
                        g.StimOff = stimOff
                    #compute correlation of responses and stimulus regressors
                    step_start1 = g.FramesPre + g.FramesStim
                    step_end1 = step_start1 + g.FramesPost
                    step_start2 = step_end1 + step_start1
                    step_end2 = step_start2 + g.FramesPost
                    con1 = np.corrcoef(g.StimOn[step_start1:],g.RawTimeseries[step_start1:step_end1])[0,1]
                    con2 = np.corrcoef(g.StimOn[step_start1:],g.RawTimeseries[step_start2:step_end2])[0,1]
                    g.CorrOn = (con1 + con2) / 2
                    cof1 = np.corrcoef(g.StimOff[step_start1:],g.RawTimeseries[step_start1:step_end1])[0,1]
                    cof2 = np.corrcoef(g.StimOff[step_start1:],g.RawTimeseries[step_start2:step_end2])[0,1]
                    g.CorrOff = (cof1 + cof2) / 2
                    #compute fourier transform on the averaged timeseries
                    ComputeFourierAvgStim(g,g.FramesPre+g.FramesFFTGap,g.FramesPre+g.FramesStim,g.StimFrequency,g.FrameRate,True)
                    #compute stimulus induced increases in calcium fluctuations
                    g.StimIndFluct = ComputeStimulusEffect(g,g.RawTimeseries)
                    #create shuffles
                    g.ComputeGraphShuffles(n_shuffles)
                    sh_mc = np.zeros(n_shuffles)#motor correlations
                    sh_con = np.zeros_like(sh_mc)#ON correlations
                    sh_coff = np.zeros_like(sh_mc)#OFF correlations
                    sh_StInFl = np.zeros_like(sh_mc)#stimulus modulation
                    sh_mfrac = np.zeros_like(sh_mc)#magnitude fraction
                    for i,row in enumerate(g.shuff_ts):
                        sh_mc[i] = np.corrcoef(row,g.PerFrameVigor)[0,1]
                        con1 = np.corrcoef(g.StimOn[step_start1:],row[step_start1:step_end1])[0,1]
                        con2 = np.corrcoef(g.StimOn[step_start1:],row[step_start2:step_end2])[0,1]
                        sh_con[i] = (con1 + con2) / 2
                        cof1 = np.corrcoef(g.StimOff[step_start1:],row[step_start1:step_end1])[0,1]
                        cof2 = np.corrcoef(g.StimOff[step_start1:],row[step_start2:step_end2])[0,1]
                        sh_coff[i] = (cof1 + cof2) / 2
                        sh_StInFl[i] = ComputeStimulusEffect(g,row)
                        avg = ComputeAveragedTimeseries(row)
                        sh_mfrac[i] = ComputeTraceFourierFraction(avg,g.FramesPre+g.FramesFFTGap,g.FramesPre+g.FramesStim,g.StimFrequency,g.FrameRate,True)
                    g.sh_m_MotorCorrelation = np.mean(sh_mc)
                    g.sh_std_MotorCorrelation = np.std(sh_mc)
                    g.sh_m_CorrOn = np.mean(sh_con)
                    g.sh_std_CorrOn = np.std(sh_con)
                    g.sh_m_CorrOff = np.mean(sh_coff)
                    g.sh_std_CorrOff = np.std(sh_coff)
                    g.sh_m_StIndFluct = np.nanmean(sh_StInFl)
                    g.sh_std_StIndFluct = np.nanstd(sh_StInFl)
                    g.sh_m_mfrac = np.mean(sh_mfrac)
                    g.sh_std_mfrac = np.std(sh_mfrac)
                    
                graph_list += graphs
        finally:
            f.close()

    non_mot_units = []#units that don't pass the motor correlation threshold
    motor_units = []#units that pass the motor correlation threshold

    #identify motor units and restrict further analysis to non-motor units
    for g in graph_list:
        if g.MotorCorrelation <= 0.5:
            non_mot_units.append(g)
        else:
            motor_units.append(g)

    #call units that show significant modulation in their standard deviation
    #as well as significant locking to our sign-wave potential stimulus units
    pot_stim_units = [g for g in non_mot_units if (g.StimIndFluct > g.sh_m_StIndFluct+2*g.sh_std_StIndFluct) and (g.mfrac_atStim > g.sh_m_mfrac + 2*g.sh_std_mfrac)]

    #identify on and off graphs based on respective correlation > 0.5
    graph_on = [g for g in pot_stim_units if g.CorrOn>g.sh_m_CorrOn+2*g.sh_std_CorrOn and g.CorrOn>0.4]
    graph_off = [g for g in pot_stim_units if g.CorrOff>g.sh_m_CorrOff+2*g.sh_std_CorrOff and g.CorrOff>0.4]

    #create plot of average per-frame-swim vigor across experiments - count each plane only once
    skip_dic = dict()
    all_vigors = []
    for g in graph_list:
        if g.SourceFile in skip_dic:
            continue
        else:
            l = g.PerFrameVigor.size
            avg_vig = ComputeAveragedTimeseries(g.PerFrameVigor)
            all_vigors.append(avg_vig)
            skip_dic[g.SourceFile] = 1
    all_vigors = np.vstack(all_vigors)

    ##manually remove graphs on eyes, etc.
    #from matplotlib.pyplot import pause
    #keep_on = np.zeros(len(graph_on))
    #skip_dic = dict()
    #for i,g in enumerate(graph_on):
    #    if g.SourceFile in skip_dic:
    #        continue
    #    PlotROI(g)
    #    pause(0.5)
    #    inp = ""
    #    while inp!="y" and inp!="n" and inp!="s" and inp!="p":
    #        inp = input("Keep unit?[y]es/[n]o/[s]kip plane/sto[p]:")
    #    if inp=="y":
    #        keep_on[i] = 1
    #    elif inp=="p":
    #        break
    #    elif inp=="s":
    #        skip_dic[g.SourceFile] = 1
    #    pl.close('all')
    #    print(i/len(graph_on),flush=True)

    #keep_off = np.zeros(len(graph_off))
    #skip_dic = dict()
    #for i,g in enumerate(graph_off):
    #    if g.SourceFile in skip_dic:
    #        continue
    #    PlotROI(g)
    #    pause(0.5)
    #    inp = ""
    #    while inp!="y" and inp!="n" and inp!="s" and inp!="p":
    #        inp = input("Keep unit?[y]es/[n]o/[s]kip plane/sto[p]:")
    #    if inp=="y":
    #        keep_off[i] = 1
    #    elif inp=="p":
    #        break
    #    elif inp=="s":
    #        skip_dic[g.SourceFile] = 1
    #    pl.close('all')
    #    print(i/len(graph_off),flush=True)

    #graph_on = [g for i,g in enumerate(graph_on) if keep_on[i]==1]
    #graph_off = [g for i,g in enumerate(graph_off) if keep_off[i]==1]


    #prepare some overview plots, using only graphs selected based on step responses
    all_on = np.vstack([ComputeDFF(g.AveragedTimeseries) for g in graph_on])
    all_off = np.vstack([ComputeDFF(g.AveragedTimeseries) for g in graph_off])
    phases_on = np.hstack([g.ang_atStim for g in graph_on])/np.pi*5+5
    phases_off = np.hstack([g.ang_atStim for g in graph_off])/np.pi*5+5

    with sns.axes_style('whitegrid'):
        pl.figure();
        sns.kdeplot(phases_on,label='On',cut=0)
        sns.kdeplot(phases_off,label='Off',cut=0)
        pl.xlim(0,10)
        pl.xlabel('Phase of response at 0.1 Hz')
        pl.ylabel('Density')
        pl.legend()
        pl.title('Phase of response to sine stimulus for ON/OFF units')

    pl.figure()
    sns.heatmap(all_off[np.argsort([g.CorrOff for g in graph_off]),:],vmin=-1,vmax=1,center=0,xticklabels=48,yticklabels=25)
    pl.xlabel('Frames - 48=20s')
    pl.ylabel('Ind. units')
    pl.title('OFF or cold-sensitive units')

    pl.figure()
    sns.heatmap(all_on[np.argsort([g.CorrOn for g in graph_on]),:],vmin=-1,vmax=1,center=0,xticklabels=48,yticklabels=25)
    pl.xlabel('Frames - 48=20s')
    pl.ylabel('Ind. units')
    pl.title('ON or heat-sensitive units')

    with sns.axes_style('whitegrid'):
        pl.figure()
        pl.plot(np.mean(all_on,0),label='ON')
        pl.plot(np.mean(all_off,0),label='OFF')
        pl.xlabel('Frames - 48=20s')
        pl.ylabel('Avg. deltaF/F')
        pl.legend()
        pl.title('Average response of ON (heat-sens.) and OFF (cold-sens.) units')

    exp_stim = np.zeros_like(graph_on[0].RawTimeseries)[:-1]
    exp_stim[72:72+144] = np.sin(np.arange(144)/2.4*2*np.pi*0.1)*300 + 700;
    exp_stim[72+144+36:288] = 700
    exp_stim[288+36:288+72] = 700
    exp_stim[72+144+180:] = exp_stim[:72+144+180]

    with sns.axes_style('whitegrid'):
        pl.figure()
        pl.plot(exp_stim,'k')
        pl.xlabel('Frames - 48=20s')
        pl.ylabel('Stimulus laser [mA]')
        pl.title('Paradigm for each imaging plane')

    #plot average swim vigor across planes

    #load taildata
    taildata = dict()
    for g in graph_list:
        if g.SourceFile in taildata:
            continue
        else:
            taildata[g.SourceFile] = LoadGraphTailData(g)

    #sort taildata: NOTE: Currently sorts all Z-Planes from different experiments together
    def ZPIndex(fname):
        try:
            ix = int(fname[-8:-6])
        except ValueError:
            ix = int(fname[-7:-6])
        return ix

    std = [taildata[k] for k in sorted(taildata.keys(), key=ZPIndex)]

    #create tail-vigor matrix for each plane - note: These won't align perfectly, so trim...
    vig_lens = [t.vigor[t.scanFrame!=-1].size for t in std]
    min_len = min(vig_lens)
    vigor_mat = np.zeros((len(std),min_len))
    for i,s in enumerate(std):
        vigor_mat[i,:] = s.vigor[:min_len]

    with sns.axes_style('white'):
        fig, ax = pl.subplots()
        m_vig = gaussian_filter1d(np.mean(mb.Bin1D(vigor_mat.copy(),2,axis=1),0),10)
        time = np.linspace(0,m_vig.size/50,m_vig.size)
        ax.plot(time,m_vig)
        ax.plot(np.arange(phase.size)/2.4,phase*3)
        ax.set_xlim(0,time.max())
        ax.set_ylabel('Average swim vigor')
        ax.set_xlabel('Time [s]')
        sns.despine()

    #create boxplot of stim and rest bout-frequencies
    f_rest = np.array([restFreq(t) for t in std])
    f_stim = np.array([stimFreq(t) for t in std])
    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        sns.boxplot(data=(f_rest,f_stim))

    #movement triggered average
    mta = ComputeSwimTriggeredAverage(motor_units,taildata,5,24)
    mta_time = np.arange(-5,25)/2.4
    with sns.axes_style('whitegrid'):
        sns.tsplot(data=mta,time=mta_time)
        pl.xlabel('Time around bout [s]');
        pl.ylabel('deltaF/F')







    ##check all
    #from matplotlib.pyplot import pause
    #skip_dic = dict()
    #for i,g in enumerate(graph_list):
    #    if g.SourceFile in skip_dic:
    #        continue
    #    PlotGraphInfo(g,i)
    #    pause(0.5)
    #    inp = ""
    #    while inp!="s" and inp!="p" and inp!="c":
    #        inp = input("[c]ontinue/[s]kip plane/sto[p]:")
    #    if inp=="p":
    #        break
    #    elif inp=="s":
    #        skip_dic[g.SourceFile] = 1
    #    pl.close('all')
    #    print(i/len(graph_list),flush=True)