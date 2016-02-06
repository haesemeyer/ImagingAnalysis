from collections import deque
import numpy as np
from PIL import Image

import matplotlib.pyplot as pl
import seaborn as sns
from scipy.stats import poisson
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d

try:
    import Tkinter
    import tkFileDialog
except ImportError:
    import tkinter as Tkinter
    import tkinter.filedialog as tkFileDialog





class CorrelationGraph:

    def __init__(self,id,timeseries):
        self.V = []#list of vertices
        self.ID = id#id of this graph for easy component reference
        self.Timeseries = timeseries#the summed pixel-timeseries of the graph

    @property
    def NPixels(self):
        return len(self.V)
#class CorrelationGraph



class PixelGraph:

    def __init__(self,id):
        self.V = []#list of vertices
        self.ID = id#id of this graph for easy component reference

    @property
    def NPixels(self):
        return len(self.V)
#class PixelGraph






def UiGetFile(filetypes = [('Tiff stack', '.tif;.tiff')],multiple=False):
    """
    Shows a file selection dialog and returns the path to the selected file(s)
    """
    options = {}
    options['filetypes'] = filetypes
    options['multiple'] = multiple
    Tkinter.Tk().withdraw() #Close the root window
    return tkFileDialog.askopenfilename(**options)
#UiGetFile


def OpenStack(filename):
    """
    Load image stack from tiff-file
    """
    im = Image.open(filename)
    stack = np.empty((im.n_frames,im.size[0],im.size[1]))
    #loop over frames and assign
    for i in range(im.n_frames):
        im.seek(i)
        stack[i,:,:] = np.array(im)
    im.close()
    return stack
    

def FilterStack(stack):
    """
    8-connected neighborhood filter of pixel timeseries. Pixel itself
    contributes 1/2 each connected neighbor 1/16
    """
    width = stack.shape[1]
    height = stack.shape[2]
    for y in range(1,height-1):
        for x in range(1,width-1):
            trace = np.zeros(stack.shape[0])
            for jit_x in range(-1,2):
                for jit_y in range(-1,2):
                    if(jit_x==0 and jit_y==0):
                        trace = trace + 0.5 * stack[:,x,y]
                    else:
                        trace = trace + 1/16 * stack[:,x+jit_x,y+jit_y]
            stack[:,x,y] = trace
    return stack#should be in place anyways

def FilterStackGaussian(stack,sigma=1):
    """
    Performs per-plane gaussian filter (assumed axis0=time)
    using the given standard deviation in pixels
    """
    out = np.zeros_like(stack,dtype='float')
    for t in range(stack.shape[0]):
        out[t,:,:] = gaussian_filter(stack[t,:,:].astype(float),sigma,multichannel=False)
    return out
    
def DeltaFOverF(stack):
    """
    Transforms stack into delta-f over F metric
    """
    F = np.mean(stack,0)[None,:,:]
    F.repeat(stack.shape[0],0)
    return (stack-F)/F
    
def Per_PixelFourier(stack,tstart=None,tend=None):
    """
    For reach pixel performs a fourier transform of the time-series (axis 0)
    between the given start and end frames. Resulting stack will have fourier
    components in axis 0, replacing time components.
    """
    if tstart is None:
        tstart = 0
    if tend is None:
        tend = stack.shape[0]#tend is exclusive
    if tend <= tstart:
        raise ValueError("tend has to be larger than tstart")
    flen = (tend-tstart)//2 + 1
    f_mag_stack = np.zeros((flen,stack.shape[1],stack.shape[2]))
    f_phase_stack = np.zeros_like(f_mag_stack)
    for y in range(stack.shape[2]):
        for x in range(stack.shape[1]):
            if(np.sum(np.isnan(stack[:,x,y]))==0):
                #compute transform on mean-subtracted trace - don't want to punish thresholding
                #for bright pixels
                transform = np.fft.rfft(stack[tstart:tend,x,y]-np.mean(stack[tstart:tend,x,y]))
                f_mag_stack[:,x,y] = np.absolute(transform)
                f_phase_stack[:,x,y] = np.angle(transform)
            else:
                f_mag_stack[:,x,y] = np.full(flen,np.NaN)
                f_phase_stack[:,x,y] = np.full(flen,np.NaN)
    return f_mag_stack, f_phase_stack
    
def ZScore_Stack(stack):
    """
    Replaces each pixel-timeseries/frequency-series with its
    corresponding z-score
    """
    avg = np.mean(stack,0)[None,:,:]
    std = np.std(stack,0)[None,:,:]
    return (stack-avg.repeat(stack.shape[0],0))/std.repeat(stack.shape[0],0)

def Threshold_Zsc(stack, nstd, plane):
    """
    Takes the ZScore of stack along axis 0 and thresholds the indicated plane
    such that all pixels above nstd will be equal to the zscore all others 0.
    The thresholded single plane is returned
    """
    #TODO: Since discrete FFT will smear out signals should probably allow passing multiple planes
    zsc = ZScore_Stack(stack)
    im_th = zsc[plane,:,:]
    im_th[im_th<nstd] = 0
    return im_th



def ConnectedComponents(image):
    """
    Using breadth-first search builds connected component graphs
    of all non-0 pixel regions
        image: The image to search
        RETURNS:
            [0]: A list of PixelGraphs, one entry for reach connected component
            [1]: A int32 image of same dimensions of image with each pixel
                labeled according to the id of the component graph it belongs to
    """
    def BFS(image,visited,sourceX,sourceY,color):
        """
        Performs breadth first search on image
        given (sourceX,sourceY) as starting pixel
        coloring all visited pixels in color
        """
        pg = PixelGraph(color)
        Q = deque()
        Q.append((sourceX,sourceY))
        visited[sourceX,sourceY] = color#mark source as visited
        while len(Q) > 0:
            v = Q.popleft()
            x = v[0]
            y = v[1]
            pg.V.append(v)#add current vertex to pixel graph
            #add non-visited neighborhood to queue
            for xn in range(x-1,x+2):#x+1 inclusive!
                for yn in range(y-1,y+2):
                    if xn<0 or yn<0 or xn>=image.shape[0] or yn>=image.shape[1]:#outside image dimensions
                        continue;
                    if (not visited[xn,yn]) and image[xn,yn]>0:
                        Q.append((xn,yn))#add non-visited above threshold neighbor
                        visited[xn,yn] = color#mark as visited
        return pg

    visited = np.zeros_like(image,dtype=int)#indicates visited pixels > 0
    conn_comps = []#list of pixel graphs
    #loop over pixels and initiate bfs whenever we encouter
    #a pixel that is non-zero and which has not yet been visited
    curr_color = 1#id counter of connected components
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (not visited[x,y]) and image[x,y]>0:
                conn_comps.append(BFS(image,visited,x,y,curr_color))
                curr_color += 1
    return conn_comps, visited

def AvgNeighbhorCorrelations(stack,dist=2,predicate=None):
    """
    Returns a 2D image which for each pixel in stack
    has the average correlation of that pixel's time-series
    with the timeseries of it's neighbors up to dist pixels
    away from the current pixel.
    Predicate is an optional function that takes an x/y coordinate
    pair as argument and returns whether to compute the correlation.
    """
    if dist<1:
        raise ValueError('Dist has to be at least 1')
    im_corr = np.zeros((stack.shape[1],stack.shape[2]))
    for x in range(stack.shape[1]):
        for y in range(stack.shape[2]):
            if (not predicate is None) and (not predicate(x,y)):#pixel is excluded
                continue
            c_sum = []
            for dx in range(-1*dist,dist+1):#dx=dist inclusive!
                for dy in range(-1*dist,dist+1):                    
                    if dx==0 and dy==0:#original pixel
                        continue
                    if x+dx<0 or y+dy<0 or x+dx>=im_corr.shape[0] or y+dy>=im_corr.shape[1]:#outside of image
                        continue
                    c_sum.append(np.corrcoef(stack[:,x,y],stack[:,x+dx,y+dy])[0,1])
            if len(c_sum)>0:
                im_corr[x,y] = np.nanmean(c_sum)
    im_corr[np.isnan(im_corr)] = 0
    return im_corr


def CorrelationConnComps(stack,im_ncorr,corr_thresh,norm8=True):
    """
    Builds connected component graphs whereby components are
    determined based on pixel-timeseries correlations. Pixels
    with timeseries correlations >= corr_thresh are considered
    as sources for a breadth-first-search which will incor-
    porate new pixels into the graph if they are correlated
    with at least corr_thresh to the summed trace of the
    component. Every new pixel's timeseries will be added
    to the sum-trace of the component.
        stack: Image time-seried stack [t,x,y]
        im_ncorr: Image with same x,y dimensions as stack
          initialized to neighborhood correlations of each pixel
        corr_thresh: Threshold correlation for source consideration
          and pixel incorporation.
        norm8: Use 8 or 4 connected components
        RETURNS:
            List of connected component graphs
    """
    def BFS(stack,thresh,visited,sourceX,sourceY,color,norm8=True):
        """
        Performs breadth first search on image
        given (sourceX,sourceY) as starting pixel
        coloring all visited pixels in color
        """
        pg = CorrelationGraph(color,np.zeros(stack.shape[0]))
        Q = deque()
        Q.append((sourceX,sourceY))
        visited[sourceX,sourceY] = color#mark source as visited
        while len(Q) > 0:
            v = Q.popleft()
            x = v[0]
            y = v[1]
            pg.V.append(v)#add current vertex to pixel graph
            pg.Timeseries = pg.Timeseries + stack[:,x,y]#add current pixel's timeseries
            #add non-visited neighborhood to queue
            for xn in range(x-1,x+2):#x+1 inclusive!
                for yn in range(y-1,y+2):
                    if xn<0 or yn<0 or xn>=stack.shape[1] or yn>=stack.shape[2] or visited[xn,yn]:#outside image dimensions or already visited
                        continue
                    if (not norm8) and xn!=x and yn!=y:
                        continue
                    #compute correlation of considered pixel's timeseries to full graphs timeseries
                    c = np.corrcoef(pg.Timeseries,stack[:,xn,yn])[0,1]
                    if c>=thresh:
                        Q.append((xn,yn))#add non-visited above threshold neighbor
                        visited[xn,yn] = color#mark as visited
        return pg

    if im_ncorr.shape[0] != stack.shape[1] or im_ncorr.shape[1] != stack.shape[2]:
        raise ValueError("Stack width and height must be same as im_ncorr width and height")
    visited = np.zeros_like(im_ncorr,dtype=int)#indicates visited pixels > 0
    conn_comps = []#list of correlation graphs
    #at each iteration we find the pixel with the highest neighborhood correlation,
    #ignoring visited pixels, and use it as a source pixel for breadth first search
    curr_color = 1#id counter of connected components
    while np.max(im_ncorr * (visited==0)) > 0:
        (x,y) = np.unravel_index(np.argmax(im_ncorr * (visited==0)),im_ncorr.shape)
        conn_comps.append(BFS(stack,corr_thresh,visited,x,y,curr_color))
        curr_color += 1
    return conn_comps, visited