# file to analyze 2-photon imaging data from sine-on-off experiments with repeat presentation
# unit graphs should have been constructed and saved before using analyzeStack.py script

from mh_2P import OpenStack, TailData, UiGetFile, NucGraph, CorrelationGraph, SOORepeatExperiment, HeatPulseExperiment
from mh_2P import MedianPostMatch, ZCorrectTrace, TailDataDict, SLHRepeatExperiment
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

import pickle

import sys
from scipy.signal import lfilter
sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')

import mhba_basic as mb


def LoadGraphTailData(graph):
    name = graph.SourceFile[:-6]+".tail"
    return TailData.LoadTailData(name, graph.CaTimeConstant)


def ZScore(trace):
    return (trace-np.mean(trace))/np.std(trace)


def ComputeAveragedTimeseries(timeseries, n_repeats, n_hangoverFrames):
    import numpy as np
    l = timeseries.size
    # sum/mean equivalent here (anyways determined by graph size)
    return np.sum(np.reshape(timeseries[0:l-n_hangoverFrames], (n_repeats, (l-1)//n_repeats)), 0)


def ComputeSwimTriggeredAverage(graph_list, tailData, f_pre, f_post):
    mta = []
    for g in graph_list:
        td = tailData[g.SourceFile]
        if td.boutFrames is None or td.boutFrames.size == 0:
            continue
        im_len = g.RawTimeseries.size
        gts = (g.RawTimeseries-np.percentile(g.RawTimeseries, 20))/np.percentile(g.RawTimeseries, 20)
        if np.any(np.isnan(gts)) or np.any(np.isinf(gts)):
            continue
        ix = mb.IndexingMatrix(td.boutFrames, f_pre, f_post, im_len)[0]
        btrig = gts[ix]
        mta.append(np.nanmean(btrig, 0))
    return np.vstack(mta)


def CorrelateResWithMTA(graph, mta):
    rs = graph.RawTimeseries
    rs = (rs-np.mean(rs))/np.std(rs)
    mta = (mta-np.mean(mta))/np.std(mta)
    return lfilter(mta[::-1], 1, rs)


def ComputeDFF(avg_timeseries):
    f0 = np.mean(avg_timeseries[:72])
    return (avg_timeseries-f0)/f0


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


def PlotAvgDff(graph, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    dff = ComputeDFF(graph.AveragedTimeseries)
    time = np.arange(dff.size)/graph.FrameRate
    ax.plot(time, dff, 'r', label='Response')
    ax.plot(time, graph.StimOn[:dff.size]*dff.max(), label='Stimulus')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('DFF')
    ax.set_title('Repeat average activity')
    ax.legend()


def PlotMotorRelation(graph,ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    # zs_act = ZScore(graph.RawTimeseries)
    zs_act = (graph.RawTimeseries - np.percentile(graph.RawTimeseries, 20)) / np.percentile(graph.RawTimeseries, 20)
    zs_mot = ZScore(graph.PerFrameVigor)
    zs_mot = zs_mot / zs_mot.max() * zs_act.max()
    time = np.arange(zs_act.size) / graph.FrameRate
    ax.plot(time, zs_act, 'r', label='Response')
    ax.plot(time, zs_mot, label='Swim vigor', alpha=0.4)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Act: DFF / Mot: AU')
    ax.set_title('Raw activity vs. motor output')
    ax.legend()


def PlotMetrics(graph,ax=None,index=None):
    if ax is None:
        fig, ax = pl.subplots()
    ax.text(0.1, 0.8, str.format('Stimulus induction {0:.2f}', graph.StimIndFluct), transform=ax.transAxes)
    ax.text(0.1, 0.7, str.format('Motor correlation {0:.2f}', graph.MotorCorrelation), transform=ax.transAxes)
    ax.text(0.1, 0.6, str.format('ON Correlation {0:.2f}', graph.CorrOn), transform=ax.transAxes)
    ax.text(0.1, 0.5, str.format('OFF Correlation {0:.2f}', graph.CorrOff), transform=ax.transAxes)
    ax.text(0.1, 0.4, str.format('Magnitude fraction {0:.2f}', graph.mfrac_atStim), transform=ax.transAxes)
    # shuffle cutoffs
    ax.text(0.6, 0.8, str.format('{0:.2f}', graph.sh_m_StIndFluct+2*graph.sh_std_StIndFluct),
            transform=ax.transAxes, color='r')
    ax.text(0.6, 0.6, str.format('{0:.2f}', graph.sh_m_CorrOn+2*graph.sh_std_CorrOn),
            transform=ax.transAxes, color='r')
    ax.text(0.6, 0.5, str.format('{0:.2f}', graph.sh_m_CorrOff+2*graph.sh_std_CorrOff),
            transform=ax.transAxes, color='r')
    ax.text(0.6, 0.4, str.format('{0:.2f}', graph.sh_m_mfrac+2*graph.sh_std_mfrac),
            transform=ax.transAxes, color='r')

    if index is not None:
        ax.text(0.1, 0.3, str.format('Array index {0}', index), transform=ax.transAxes)
    ax.text(0.1, 0.2, str.format('Sourcefile {0}', graph.SourceFile.split('/')[-1]), transform=ax.transAxes)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def PlotGraphInfo(graph, index=None):
    with sns.axes_style('white'):
        fig, axes = pl.subplots(2, 2, figsize=(15, 10))
        PlotROI(graph, axes[0, 0])
        PlotAvgDff(graph, axes[0, 1])
        PlotMotorRelation(graph, axes[1, 0])
        PlotMetrics(graph, axes[1, 1], index)
        sns.despine(fig)
        fig.tight_layout()


def PlotCycleGraphList(glist):
    from matplotlib.pyplot import pause
    skip_dic = dict()
    for i,g in enumerate(glist):
        if g.SourceFile in skip_dic:
            continue
        PlotGraphInfo(g, i)
        pause(0.5)
        inp = ""
        while inp != "s" and inp != "p" and inp != "c":
            inp = input("[c]ontinue/[s]kip plane/sto[p]:")
        if inp == "p":
            break
        elif inp == "s":
            skip_dic[g.SourceFile] = 1
        pl.close('all')
        print(i/len(glist), flush=True)


def restFreq(tail_d):
    if tail_d.bouts is None:
        return 0
    st_b = np.sum(data.stimOn[tail_d.boutFrames.astype(int)] == 0)
    return st_b/150


def stimFreq(tail_d):
    if tail_d.bouts is None:
        return 0
    st_b = np.sum(data.stimOn[tail_d.boutFrames.astype(int)] != 0)
    return st_b/180


def tryZCorrect(graphs, eyemask=None):
    sf = graphs[0].SourceFile
    stabFile = sf.replace("_0.tif", "_stableZ_0.tif")
    try:
        stack = OpenStack(sf)
        pre = OpenStack(stabFile)
    except IOError:
        print("Could not load stableZ prestack for ", sf)
        return
    # if information is present mask out pixels on eyes
    if eyemask is not None:
        eyemask[eyemask > 0] = 1
        eyemask[eyemask <= 0] = 0
        stack = stack * eyemask
        pre = pre * eyemask
    sl = MedianPostMatch(pre, stack, interpolate=True, nIter=100)[0]
    # create slice-summed projection of pre-stack
    sum_stack = np.zeros((pre.shape[0] // 21, pre.shape[1], pre.shape[2]), dtype=np.float32)
    for i in range(sum_stack.shape[0]):
        sum_stack[i, :, :] = np.sum(pre[i * 21:(i + 1) * 21, :, :], 0)
    for g in graphs:
        g.Original = g.RawTimeseries.copy()
        g.RawTimeseries = ZCorrectTrace(sum_stack, sl, g.RawTimeseries, g.V)


def frameBoutStarts(expData):
    done = dict()
    starts = []
    for i in range(expData.Vigor.shape[0]):
        name = expData.graph_info[i][0]
        if name in done:
            continue
        done[name] = True
        tdata = tdd[name]
        bstarts = tdata.starting
        times = tdata.frameTime
        digitized = np.digitize(times, i_t)
        t = np.array([bstarts[digitized == i].sum() for i in range(1, interp_times.size + 1)])
        starts.append(t)
    return np.vstack(starts)


if __name__ == "__main__":
    interpol_freq = 5  # the frequency to which we interpolate before analysis


    n_repeats = int(input("Please enter the number of repeats:"))  # the number of repeats performed in each plane
    s_pre = int(input("Please enter the number of pre-seconds:"))
    n_pre = s_pre * interpol_freq
    s_stim = int(input("Please enter the number of stimulus-seconds:"))
    n_stim = s_stim * interpol_freq  # the number of stimulus frames at interpol_freq
    try:
        s_post = int(input("Please enter the number of post-seconds or press enter for 75:"))
    except:
        print("Assuming default value of 75s")
        s_post = 75
    n_post = s_post * interpol_freq
    t_per_frame = float(input("Please enter the duration of each frame in seconds:"))
    ans = ""

    exp_type = int(input("Experiment type: SOORepeat [0], LoHi [1], HeatPulse[2]"))
    if exp_type < 0 or exp_type > 2:
        raise ValueError("Invalid experiment type selected")

    ans = ""
    while ans != 'n' and ans != 'y':
        ans = input("Load eye mask file? [y/n]:")
    if ans == 'n':
        eyemask = None
    else:
        mfile = UiGetFile([('Eye mask file', 'EYEMASK*')])
        eyemask = OpenStack(mfile)

    n_hangoverFrames = 0  # the number of "extra" recorded frames at experiment end - = 0 after interpolation

    n_shuffles = 200
    s_amp = 0.36  # sine amplitude relative to offset

    # load unit activity from each imaging plane, compute motor correlations
    # and repeat averaged time-trace <- note that motor correlations have to
    # be calculated on non-time averaged data
    graphFiles = UiGetFile([('PixelGraphs','.graph')],multiple=True)
    graph_list = []

    # load all units from file
    for fname in graphFiles:
        f = open(fname, 'rb')
        graphs = pickle.load(f)
        if len(graphs) > 0:
            tryZCorrect(graphs, eyemask)
        graph_list += graphs

    tdd = TailDataDict(graph_list[0].CaTimeConstant)
    # add frame bout start trace to graphs if required
    # for g in graph_list:
    #     # if hasattr(g, 'BoutStartTrace') and g.BoutStartTrace is not None:
    #     #    continue
    #     tdata = tdd[g.SourceFile]
    #     g.BoutStartTrace = tdata.FrameBoutStarts

    frame_times = np.arange(graph_list[0].RawTimeseries.size) * t_per_frame
    # endpoint=False in the following call will remove hangover frame
    interp_times = np.linspace(0, (s_pre+s_stim+s_post)*n_repeats, (n_pre+n_stim+n_post)*n_repeats, endpoint=False)

    # interpolate calcium data
    ipol = lambda y: np.interp(interp_times, frame_times, y[:frame_times.size])
    interp_data = np.vstack([ipol(g.RawTimeseries) for g in graph_list])

    # interpolate convolved bout starts (means interpolating down from original 100Hz trace)
    traces = dict()
    bstarts = []
    # for binning, we want to have one more subdivision of the times and include the endpoint - later each time-bin
    # will correspond to one timepoint in interp_times above
    i_t = np.linspace(0, (s_pre+s_stim+s_post)*n_repeats, (n_pre+n_stim+n_post)*n_repeats + 1, endpoint=True)
    for g in graph_list:
        if g.SourceFile not in traces:
            tdata = tdd[g.SourceFile]
            conv_bstarts = tdata.ConvolvedStarting
            times = tdata.frameTime
            digitized = np.digitize(times, i_t)
            t = np.array([conv_bstarts[digitized == i].sum() for i in range(1, interp_times.size+1)])
            traces[g.SourceFile] = t
        bstarts.append(traces[g.SourceFile])
    interp_starts = np.vstack(bstarts)

    if exp_type == 1:
        data = SLHRepeatExperiment(interp_data,
                                   interp_starts,
                                   n_pre, n_stim, n_post, n_repeats, graph_list[0].CaTimeConstant,
                                   nHangoverFrames=0, frameRate=interpol_freq)
    elif exp_type == 0:
        data = SOORepeatExperiment(interp_data,
                                   interp_starts,
                                   n_pre, n_stim, n_post, n_repeats, graph_list[0].CaTimeConstant,
                                   nHangoverFrames=0, frameRate=interpol_freq)
    elif exp_type == 2:
        data = HeatPulseExperiment(interp_data,
                                   interp_starts,
                                   n_pre, n_stim, n_post, n_repeats, graph_list[0].CaTimeConstant,
                                   nHangoverFrames=0, frameRate=interpol_freq, baseCurrent=0)

    data.graph_info = [(g.SourceFile, g.V) for g in graph_list]
    data.original_time_per_frame = t_per_frame

    # stop

    ts_avg = data.repeatAveragedTimeseries(data.nRepeats, data.nHangoverFrames)
    motor_correlation, m_sh_mc, std_sh_mc = data.motorCorrelation(n_shuffles)
    corr_on, m_sh_con, std_sh_con = data.onStimulusCorrelation(n_shuffles)
    corr_off, m_sh_cof, std_sh_cof = data.offStimulusCorrelation(n_shuffles)
    fft, freqs, mag, mfrac, ang = data.computeFourierMetrics(True)
    stim_fluct, m_sh_sid, std_sh_sid = data.computeStimulusEffect(n_shuffles)
    m_sh_mfrac, std_sh_mfrac = data.computeFourierFractionShuffles(n_shuffles, True)
    # correlation for transient on and off
    trans_on = np.r_[0, np.diff(data.stimOn)]
    trans_on[trans_on < 0] = 0
    c_transOn = data.stimulusCorrelation(trans_on)[0]
    trans_off = np.r_[0, np.diff(data.stimOff)]
    trans_off[trans_off < 0] = 0
    c_transOff = data.stimulusCorrelation(trans_off)[0]

    # temporarily re-assign all information to graphs in order to re-use code below...
    for i, g in enumerate(graph_list):
        g.MotorCorrelation = motor_correlation[i]
        g.AveragedTimeseries = ts_avg[i, :]
        g.StimOn = data.stimOn
        g.StimOff = data.stimOff
        g.CorrOn = corr_on[i]
        g.Corr_TOn = c_transOn[i]
        g.CorrOff = corr_off[i]
        g.Corr_TOff = c_transOff[i]
        g.stimFFT = fft[i, :]
        g.stimFFT_freqs = freqs
        g.mag_atStim = mag[i]
        g.mfrac_atStim = mfrac[i]
        g.ang_atStim = ang[i]
        g.StimIndFluct = stim_fluct[i]
        # shuffles
        g.sh_m_MotorCorrelation = m_sh_mc[i]
        g.sh_std_MotorCorrelation = std_sh_mc[i]
        g.sh_m_CorrOn = m_sh_con[i]
        g.sh_std_CorrOn = std_sh_con[i]
        g.sh_m_CorrOff = m_sh_cof[i]
        g.sh_std_CorrOff = std_sh_cof[i]
        g.sh_m_StIndFluct = m_sh_sid[i]
        g.sh_std_StIndFluct = std_sh_sid[i]
        g.sh_m_mfrac = m_sh_mfrac[i]
        g.sh_std_mfrac = std_sh_mfrac[i]

    non_mot_units = []  # units that don't pass the motor correlation threshold
    motor_units = []  # units that pass the motor correlation threshold

    # identify motor units and restrict further analysis to non-motor units
    for g in graph_list:
        if g.MotorCorrelation <= 0.5:
            non_mot_units.append(g)
        else:
            motor_units.append(g)

    # call units that show significant modulation in their standard deviation
    # as well as significant locking to our sign-wave potential stimulus units
    pot_stim_units = [g for g in non_mot_units if (g.StimIndFluct > g.sh_m_StIndFluct+2*g.sh_std_StIndFluct) and
                      (g.mfrac_atStim > g.sh_m_mfrac + 2*g.sh_std_mfrac)]

    # identify on and off graphs based on respective correlation > 0.5
    graph_on = [g for g in pot_stim_units if g.CorrOn>g.sh_m_CorrOn+2*g.sh_std_CorrOn and g.CorrOn > 0.4]
    graph_off = [g for g in pot_stim_units if g.CorrOff>g.sh_m_CorrOff+2*g.sh_std_CorrOff and g.CorrOff > 0.4]

    # create plot of average per-frame-swim vigor across experiments - count each plane only once
    skip_dic = dict()
    all_vigors = []
    for g in graph_list:
        if g.SourceFile in skip_dic:
            continue
        else:
            l = g.PerFrameVigor.size
            avg_vig = ComputeAveragedTimeseries(g.PerFrameVigor, n_repeats, n_hangoverFrames)
            all_vigors.append(avg_vig)
            skip_dic[g.SourceFile] = 1
    all_vigors = np.vstack(all_vigors)


    # prepare some overview plots, using only graphs selected based on step responses
    all_on = np.vstack([ComputeDFF(g.AveragedTimeseries) for g in graph_on])
    all_off = np.vstack([ComputeDFF(g.AveragedTimeseries) for g in graph_off])
    phases_on = np.hstack([g.ang_atStim for g in graph_on])/np.pi*5+5
    phases_off = np.hstack([g.ang_atStim for g in graph_off])/np.pi*5+5

    with sns.axes_style('whitegrid'):
        pl.figure()
        sns.kdeplot(phases_on,label='On',cut=0)
        sns.kdeplot(phases_off,label='Off',cut=0)
        pl.xlim(0, 10)
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

    with sns.axes_style('whitegrid'):
        pl.figure()
        pl.plot(data.stimOn, 'k')
        pl.xlabel('Frames - 48=20s')
        pl.ylabel('Stim ON')
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
        ax.plot(np.arange(data.stimOn.size)/2.4,data.stimOn*3)
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
        pl.figure()
        sns.tsplot(data=mta,time=mta_time)
        pl.xlabel('Time around bout [s]');
        pl.ylabel('deltaF/F')
