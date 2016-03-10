# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:53:30 2016

@author: mhaesemeyer
"""

from mh_2P import *

import warnings

from scipy.ndimage.morphology import binary_fill_holes

import pickle

from tkinter import Label, Toplevel, Button, Entry, Tk



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


def AssignFourier(graph,startFrame,endFrame,suffix="pre"):
    global des_freq
    #anti-aliasing
    filtered = gaussian_filter1d(graph.RawTimeseries,1.2)
    filtered = filtered[startFrame:endFrame]
    #TODO: Somehow make the following noise reduction more generally applicable...
    #if the length of filtered is divisble by 2, break into two blocks and average for noise reduction
    if filtered.size % 2 == 0:
        filtered = np.mean(filtered.reshape((2,filtered.size//2)),0)
    fft = np.fft.rfft(filtered)
    freqs = np.linspace(0,1.2,fft.shape[0])
    mag = np.absolute(fft)
    ix = np.argmin(np.absolute(des_freq-freqs))#index of bin which contains our desired frequency
    setattr(graph,"fft_"+suffix,fft)
    setattr(graph,"freqs_"+suffix,freqs)
    setattr(graph,"fourier_ratio_"+suffix,mag[ix]/mag.sum())


#User entry dialog of scipt settings
class InputDialog:
    def __init__(self,parent):
        top = self.top = Toplevel(parent)
        Label(top,text="Pre Stim Frames").grid(row=0)
        Label(top,text="Stim Frames").grid(row=1)
        Label(top,text="Post Stim Frames").grid(row=2)
        Label(top,text="Stim fft gap").grid(row=3)
        Label(top,text="Corr threshold").grid(row=4)
        Label(top,text="Avg. cell diameter").grid(row=5)

        self.e_preStim = Entry(top)
        self.e_preStim.grid(row=0,column=1)
        self.e_preStim.insert(0,144)

        self.e_Stim = Entry(top)
        self.e_Stim.grid(row=1,column=1)
        self.e_Stim.insert(0,144)

        self.e_postStim = Entry(top)
        self.e_postStim.grid(row=2,column=1)
        self.e_postStim.insert(0,144)

        self.e_stimFftGap = Entry(top)
        self.e_stimFftGap.grid(row=3,column=1)
        self.e_stimFftGap.insert(0,48)

        self.e_corrTh = Entry(top)
        self.e_corrTh.grid(row=4,column=1)
        self.e_corrTh.insert(0,0.3)

        self.e_avgCdiam = Entry(top)
        self.e_avgCdiam.grid(row=5,column=1)
        self.e_avgCdiam.insert(0,8)

        self.b = Button(top,text="OK",command = self.ok)
        self.b.grid(row=7)

    def ok(self):
        self.pre_stim = int(self.e_preStim.get())
        self.stim = int(self.e_Stim.get())
        self.post_stim = int(self.e_postStim.get())
        self.stim_fftGap = int(self.e_stimFftGap.get())
        self.corrTh = float(self.e_corrTh.get())
        self.cellDiam = int(self.e_avgCdiam.get())
        self.top.destroy()


if __name__ == "__main__":
    #warnings.simplefilter("error",RuntimeWarning)
    save = True
    #paradigm constants
    #pre_stim = 144#144#48
    #stim = 144#48
    #post_stim = 144#144#48
    #stim_fft_gap = 24#0#number of initial frames not to use for fft determination
    min_phot = 10#minimum number of photons to observe in a pixels timeseries to not discard series
    des_freq = 0.1#0.3#1/3##the stimulus frequency
    frame_rate = 2.4# imaging frame-rate in Hz
    #corr_thresh = 0.2#0.5#0.1#0.025#correlation threshold for co-segmenting pixels - should probably be >0.5 if things actually worked

    #obtain paradigm constants from user
    root = Tk()
    diag = InputDialog(root)
    root.wait_window(diag.top)
    pre_stim = diag.pre_stim
    stim = diag.stim
    post_stim = diag.post_stim
    stim_fft_gap = diag.stim_fftGap
    corr_thresh = diag.corrTh
    cell_diam = diag.cellDiam
    root.destroy()


    filenames = UiGetFile([('Tiff Stack', '.tif;.tiff')],multiple=True)

    
    if type(filenames) is str:
        filenames = [filenames]
    for i,f in enumerate(filenames):
        #load stack - first try to find aligned file
        try:
            stack = np.load(f[:-4]+"_stack.npy").astype('float')
            print('Loaded aligned stack of slice ', i,flush=True)
        except FileNotFoundError:
            stack = OpenStack(f).astype(float)
            #re-align slices
            stack, xshift, yshift = ReAlign(stack)
            #save the re-aligned stack as uint8
            if save:
                np.save(f[:-4]+"_stack.npy",stack.astype(np.uint8))
                #save alignment shifts
                np.save(f[:-4]+"_xshift.npy",xshift.astype(np.int32))
                np.save(f[:-4]+"_yshift.npy",yshift.astype(np.int32))
            print('Stack ',i,' of ',len(filenames)-1,' realigned',flush=True)
        #perform quality control and save resulting plot
        qualscore = CorrelationControl(stack,36)[0]
        fig = pl.figure()
        pl.plot(qualscore/np.mean(qualscore),'o')
        pl.xlabel('Stack section')
        pl.ylabel('Fraction of mean correlation')
        #draw in 1.1 and 0.9 lines as potential relaxed cut-offs
        pl.plot([-0.1,qualscore.size+0.1],[1.05,1.05],'k--',alpha=0.5)
        pl.plot([-0.1,qualscore.size+0.1],[1.1,1.1],'k--')
        pl.plot([-0.1,qualscore.size+0.1],[0.95,0.95],'k--',alpha=0.5)
        pl.plot([-0.1,qualscore.size+0.1],[0.9,0.9],'k--')
        sns.despine()
        if save:
            fig.savefig(f[:-4]+'_qualscores.pdf',type='pdf')
            pl.close('all')

        #create shuffled stack to determine correlation seed cut-off
        st_shuff = ShuffleStackTemporal(stack)
        #we only want to consider time-series with at least min_phot photons
        sum_stack = np.sum(stack,0)
        consider = lambda x,y: sum_stack[x,y]>=min_phot
        #compute photon-rates using gaussian windowing - since we also filter spatially, needs to be done BEFORE zscoring!!!
        rate_stack = gaussian_filter(stack,(frame_rate,cell_diam/8,cell_diam/8))#along time standard deviation of 1s, 1/8 of cell diameter along spatial dimension - i.e. filter drops to ~0 after cell radius
        rs_shuff = gaussian_filter(st_shuff,(frame_rate,cell_diam/8,cell_diam/8))
        
        #compute neighborhood correlations of pixel-timeseries for segmentation seeds
        im_ncorr = AvgNeighbhorCorrelations(rate_stack,2,consider)
        im_nc_shuff = AvgNeighbhorCorrelations(rs_shuff,2,consider)
        print('Maximum neighbor correlation in stack ',i,' = ',im_ncorr.max(),flush=True)

        #determine correlation seed cutoff - find correlation value where correlations larger that value
        #are enriched at least ten times in the real dataset over the shuffled data-set
        seed_cutoff = 1
        for c in np.linspace(0,1,1001):
            if ((im_ncorr>c).sum() / (im_nc_shuff>c).sum()) >= 10:
                seed_cutoff = c
                break
        print('Correlation seed cutoff in stack ',i,' = ',seed_cutoff,flush=True)

        #extract correlation graphs - 4-connected
        #cap our growth correlation threshold at the seed-cutoff, i.e. if corr_thresh
        #is larger than the significance threshold reduce it, when creating graph
        if corr_thresh <= seed_cutoff:
            ct_actual = corr_thresh
        else:
            ct_actual = seed_cutoff
        graph, colors = CorrelationGraph.CorrelationConnComps(rate_stack,im_ncorr,ct_actual,False,(0,rate_stack.shape[0]),seed_cutoff)
        print('Correlation graph of stack ',i,' of ',len(filenames)-1,' created',flush=True)
        
        #plot largest three components onto projection
        #projection = np.zeros((im_ncorr.shape[0],im_ncorr.shape[1],3),dtype=float)
        #projection[:,:,0] = projection[:,:,2] = sum_stack / sum_stack.max()
        #g_sizes = [g.NPixels for g in graph]
        #g_sizes = sorted(g_sizes)
        #for g in graph:
        #    if g.NPixels > g_sizes[-4]:
        #        for v in g.V:
        #            projection[v[0],v[1],1] = 0.3#color it green
        #with sns.axes_style('white'):
        #    fig, ax = pl.subplots()
        #    ax.imshow(projection)

        max_qual_deviation = (np.max(qualscore)-np.min(qualscore))/np.mean(qualscore)
        print("Maximum quality score deviation = ",max_qual_deviation,flush=True)

        #TODO: clean up graphs - potentially "fill holes" and remove very small connected components
        graph = [g for g in graph if g.NPixels>=30]#remove compoments with less than 30 pixels
        print('Identified ',len(graph),'units in slice ',i,flush=True)
        #for each graph, create sum-trace which is only lightly filtered
        #assign to graph object and assign fft fraction at desired frequency
        #to pre, stim and post
        for g in graph:
            g.SourceFile = f#store for convenience access
            g.FramesPre = pre_stim
            g.FramesStim = stim
            g.FramesPost = post_stim
            g.FramesFFTGap = stim_fft_gap
            g.StimFrequency = des_freq
            g.CellDiam = cell_diam
            g.CorrThresh = ct_actual
            g.CorrSeedCutoff = seed_cutoff
            g.RawTimeseries = np.zeros_like(g.Timeseries)
            g.FrameRate = frame_rate
            #also save in graph the maximum absolute quality score deviation from the mean
            g.MaxQualScoreDeviation = max_qual_deviation
            for v in g.V:
                g.RawTimeseries = g.RawTimeseries + stack[:,v[0],v[1]]
            #pre-stim
            AssignFourier(g,stim_fft_gap,pre_stim,"pre")
            #stim
            AssignFourier(g,pre_stim+stim_fft_gap,pre_stim+stim,"stim")
            #post-stim
            AssignFourier(g,pre_stim+stim+stim_fft_gap,pre_stim+stim+post_stim,"post")

        #save graph list
        if save:
            f_graph = open(f[:-3]+"graph","wb")
            pickle.dump(graph,f_graph,protocol=pickle.HIGHEST_PROTOCOL)
            f_graph.close()
        #list all graphs in which the fourier ratio during stimulation compared to
        #pre and post has increased by at least 0.1 on average
        inc = lambda gr: np.mean([gr.fourier_ratio_stim-gr.fourier_ratio_pre,gr.fourier_ratio_stim-gr.fourier_ratio_post])
        power_inc = [(i,inc(g)) for i,g in enumerate(graph) if inc(g)>0.05 and g.fourier_ratio_stim > 0.1]
        if power_inc:
            power_inc, increases = zip(*power_inc)
        else:
            power_inc = increases = []

        #draw all graphs into the image and save
        projection = np.zeros((im_ncorr.shape[0],im_ncorr.shape[1],3),dtype=float)
        projection[:,:,0] = projection[:,:,2] = sum_stack / sum_stack.max()
        im_roi = np.zeros_like(im_ncorr)
        for g in graph:
            for v in g.V:
                im_roi[v[0],v[1]] = 1
        im_roi = binary_fill_holes(im_roi)
        projection[:,:,1] = im_roi
        im_roi = Image.fromarray((projection*255).astype(np.uint8))
        if save:
            im_roi.save(f[:-3]+'png')

        #plot trace of graph in this plane with maximum increase
        if len(graph) > 0 and len(power_inc) > 0:
            frames = np.arange(pre_stim+stim+post_stim)
            m_graph = graph[power_inc[np.argmax(increases)]]
            with sns.axes_style('white'):
                fig, (ax1, ax2, ax3) = pl.subplots(ncols=3)
                ax1.plot(frames[:pre_stim],m_graph.RawTimeseries[:pre_stim])
                ax1.plot(frames[pre_stim+stim:pre_stim+stim+post_stim],m_graph.RawTimeseries[pre_stim+stim:pre_stim+stim+post_stim])
                ax1.plot(frames[pre_stim:pre_stim+stim],m_graph.RawTimeseries[pre_stim:pre_stim+stim])
                ax1.plot([pre_stim,pre_stim],[ax1.get_ylim()[0],ax1.get_ylim()[1]],'k--',alpha=0.5)
                ax1.plot([pre_stim+stim,pre_stim+stim],[ax1.get_ylim()[0],ax1.get_ylim()[1]],'k--',alpha=0.5)
                ax1.set_xlabel('Frames')
                ax1.set_ylabel('ZScored photon count')
                ax2.plot(m_graph.freqs_pre,np.absolute(m_graph.fft_pre**2),label="Pre")
                ax2.plot(m_graph.freqs_post,np.absolute(m_graph.fft_post**2),label="Post")
                ax2.plot(m_graph.freqs_stim,np.absolute(m_graph.fft_stim**2),label="Stim")
                ax2.plot([des_freq,des_freq],[0,ax2.get_ylim()[1]],'k--',alpha=0.5)
                ax2.set_xlabel('Frequency [Hz]')
                ax2.set_ylabel('Power')
                ax2.legend()
                projection[:,:,0] = projection[:,:,1] = projection[:,:,2] = sum_stack/sum_stack.max()*2
                projection[projection>0.8] = 0.8
                for v in m_graph.V:
                    projection[v[0],v[1],0] = 1
                ax3.imshow(projection)
                sns.despine(ax=ax3,left=True,bottom=True)
                fig.set_size_inches(15,5,5)
                fig.tight_layout()
            if save:
                fig.savefig(f[:-3]+'pdf',type='pdf')
                pl.close('all')
            

        #plot pre, stim and post fourier magnitude
        

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