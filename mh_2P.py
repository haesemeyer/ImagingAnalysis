from collections import deque
import numpy as np
from PIL import Image

import matplotlib.pyplot as pl
import seaborn as sns
from scipy.stats import poisson
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from scipy.signal import fftconvolve

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

    @property
    def MinX(self):
        return min(list(zip(*self.V))[0])

    @property
    def MinY(self):
        return min(list(zip(*self.V))[1])

    @property
    def MaxX(self):
        return max(list(zip(*self.V))[0])

    @property
    def MaxY(self):
        return max(list(zip(*self.V))[1])

    @staticmethod
    def CorrelationConnComps(stack,im_ncorr,corr_thresh,norm8=True,limit=None,seed_limit=0):
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
            limit: Optionally specify start and end of timeslice over
              which correlations should be computed
            seed_limit: For a pixel to be considered a valid seed, it's
              neighborhood correlation needs to exceed this value
            RETURNS:
                [0]: List of connected component graphs
                [1]: Image numerically identifying each pixel of each graph
        """
        def BFS(stack,thresh,visited,sourceX,sourceY,color,norm8):
            """
            Performs breadth first search on image
            given (sourceX,sourceY) as starting pixel
            coloring all visited pixels in color
            """
            nonlocal limit
            pg = CorrelationGraph(color,np.zeros(stack.shape[0]))
            Q = deque()
            Q.append((sourceX,sourceY,0))
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
                        c = np.corrcoef(pg.Timeseries[limit[0]:limit[1]],stack[limit[0]:limit[1],xn,yn])[0,1]
                        if c>=thresh:
                            Q.append((xn,yn,v[2]+1))#add non-visited above threshold neighbor
                            visited[xn,yn] = color#mark as visited
            return pg

        if im_ncorr.shape[0] != stack.shape[1] or im_ncorr.shape[1] != stack.shape[2]:
            raise ValueError("Stack width and height must be same as im_ncorr width and height")
        if limit is None:
            limit = (0,stack.shape[0])
        if len(limit) != 2:
            raise ValueError("If set limit must be a 2-element tuple or list referencing start and end slice (exclusive)")
        visited = np.zeros_like(im_ncorr,dtype=int)#indicates visited pixels > 0
        conn_comps = []#list of correlation graphs
        #at each iteration we find the pixel with the highest neighborhood correlation,
        #ignoring visited pixels, and use it as a source pixel for breadth first search
        curr_color = 1#id counter of connected components
        while np.max(im_ncorr * (visited==0)) > seed_limit:
            (x,y) = np.unravel_index(np.argmax(im_ncorr * (visited==0)),im_ncorr.shape)
            conn_comps.append(BFS(stack,corr_thresh,visited,x,y,curr_color,norm8))
            curr_color += 1
        return conn_comps, visited
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
    #do not introduce NaN's in zero rows (which will have avg=std=0) but rather keep
    #them as all zero
    std[avg==0] = 1
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
    corr_buff = dict()#buffers computed correlations to avoid computing the same pairs multiple times!
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
                    p_src = (x,y)
                    p_des = (x+dx,y+dy)
                    if (p_src,p_des) in corr_buff:
                        c_sum.append(corr_buff[(p_src,p_des)])
                    else:
                        cval = np.corrcoef(stack[:,x,y],stack[:,x+dx,y+dy])[0,1]
                        corr_buff[(p_des,p_src)] = cval
                        c_sum.append(cval)
            if len(c_sum)>0 and not np.all(np.isnan(c_sum)):
                im_corr[x,y] = np.nanmean(c_sum)
    im_corr[np.isnan(im_corr)] = 0
    return im_corr


def ComputeAlignmentShift(stack, index):
    """
    For the slice in stack identified by index computes
    the x (row) and y (column) shift that corresponds to
    the best alignment of stack[index,:,:] to the re-
    mainder of the stack
    """
    #shift_x = np.zeros(stack.shape[0])#best x-shift of each image
    #shift_y = np.zeros_like(shift_x)#best y-shift of each image
    #max_corr = np.zeros_like(shift_x)#un-normalized correlation at best shift

    if index == 0:
        sum_stack = np.sum(stack[1:,:,:],0)
    elif index == stack.shape[0]-1:
        sum_stack = np.sum(stack[:-1,:,:],0)
    else: 
        sum_stack = np.sum(stack[:index,:,:],0) + np.sum(stack[index+1:,:,:],0)

    exp_x, exp_y = stack.shape[1]-1, stack.shape[2]-1#these are the indices in the cross-correlation matrix that correspond to 0 shift    
    c = fftconvolve(sum_stack,stack[index,::-1,::-1])
    x,y = np.unravel_index(np.argmax(c),c.shape)
    shift_x = x-exp_x
    shift_y = y-exp_y
    return shift_x,shift_y

def ReAlign(stack):
    """
    Re-positions every slice in stack by the following iterative
    procedure:
    Compute sum over stack excluding current slice
    Compute best-aligned x-shift and y-shift of current slice to sum-stack
    Shift the slice by x-shift and y-shift within the stack
    Increment current slice and repeat from top
    """
    def Shift2Index(shift,size):
        """
        Translates a given shift into the appropriate
        source and target indices
        """
        if shift<0:
            #coordinate n in source should be n-shift in target
            source = (-1*shift,size)
            target = (0,shift)
        elif shift>0:
            #coordinate n in source should be n+1 in target
            source = (0,-1*shift)
            target = (shift,size)
        else:
            source = (0,size)
            target = (0,size)
        return source, target
    x_shifts = np.zeros(stack.shape[0])
    y_shifts = np.zeros_like(x_shifts)
    re_aligned = stack.copy()
    for t in range(re_aligned.shape[0]):
        xshift, yshift = ComputeAlignmentShift(re_aligned,t)
        x_shifts[t] = xshift
        y_shifts[t] = yshift
        if xshift == 0 and yshift == 0:
            continue
        if np.abs(xshift)>3 or np.abs(yshift)>3:
            print("Warning. Slice ",t," requires shift greater 3 pixels. Not shifted")
            continue
        xs,xt = Shift2Index(xshift,re_aligned.shape[1])
        ys,yt = Shift2Index(yshift,re_aligned.shape[2])
        newImage = np.zeros((re_aligned.shape[1],re_aligned.shape[2]))
        newImage[xt[0]:xt[1],yt[0]:yt[1]] = re_aligned[t,xs[0]:xs[1],ys[0]:ys[1]]
        re_aligned[t,:,:] = newImage
    return re_aligned, x_shifts, y_shifts

def ShuffleStackSpatioTemporal(stack):
    """
    Returns a version of stack that has been randomly shuffled
    along it's spatial dimensions (axis 1 and 2) as well as
    circularly permuted along it's temporal dimension (axis 0)
    """
    shuff = stack.copy()
    s0,s1,s2 = stack.shape
    for x in range(s1):
        for y in range(s2):
            xs = np.random.randint(s1)
            ys = np.random.randint(s2)
            temp = shuff[:,xs,ys].copy()#this copy is important since otherwise we are dealing with a view not actual values!!!
            shuff[:,xs,ys] = shuff[:,x,y]
            shuff[:,x,y] = np.roll(temp,np.random.randint(s0))
    return shuff

def ShuffleStackTemporal(stack):
    """
    Returns a version of the stack that has been
    permuted at random along the time axis (axis 0)
    """
    shuff = np.empty_like(stack)
    s0,s1,s2 = stack.shape
    for x in range(s1):
        for y in range(s2):
            shuff[:,x,y] = np.random.choice(stack[:,x,y],size=s0,replace=False)
    return shuff
            