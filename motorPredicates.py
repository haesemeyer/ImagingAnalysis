# Functions that classify detected bouts based on their characteristics
import numpy as np
from scipy.stats import mode
from mh_2P import TailData

strong_thresh = 0.8
turn_thresh = 0.5


def safe_set(array, index, value):
    """
    Sets and element in an array if it exists
    Args:
        array: The array in which to set an element
        index: The index of the element
        value: The new value

    Returns:
        True if element was set, false otherwise
    """
    if index >= array.size:
        return False
    array[index] = value
    return True


def bias(start, end, ca, mca):
    """
    For a bout computes it's bias i.e. how much of the stroke is in one vs. the other direction
    Args:
        start: The start frame of the bout
        end: The end frame of the bout
        ca: The cumulative angle trace for the whole experiment
        mca: A baseline value to be subtracted before calculating the bias

    Returns:
        A value btw. -1 and 1 reflecting the preference for positive vs. negative cumulative angles within the bout
    """
    bca = ca[start:end+1] - mca
    pos = bca[bca > 0]
    neg = bca[bca < 0]
    return (pos.sum()+neg.sum())/(pos.sum()-neg.sum())


def left_bias_bouts(tdata: TailData):
    """
    For a given TailData object returns a bout start array for bouts that are strongly left biased
    """
    if tdata.bouts is None:
        return tdata.starting
    starting = np.zeros(tdata.starting.size, dtype=np.float32)
    mca = mode(tdata.cumAngles)[0]
    for b in tdata.bouts.astype(int):
        bb = bias(b[0], b[1], tdata.cumAngles, mca)
        if bb <= -strong_thresh:
            safe_set(starting, b[0], 1)
    return starting


def right_bias_bouts(tdata: TailData):
    """
        For a given TailData object returns a bout start array for bouts that are strongly right biased
    """
    if tdata.bouts is None:
        return tdata.starting
    starting = np.zeros(tdata.starting.size, dtype=np.float32)
    mca = mode(tdata.cumAngles)[0]
    for b in tdata.bouts.astype(int):
        bb = bias(b[0], b[1], tdata.cumAngles, mca)
        if bb >= strong_thresh:
            safe_set(starting, b[0], 1)
    return starting


def high_bias_bouts(tdata: TailData):
    """
    For a given TailData object returns a bout start array for bouts that are strongly biased in either direction
    """
    if tdata.bouts is None:
        return tdata.starting
    starting = np.zeros(tdata.starting.size, dtype=np.float32)
    mca = mode(tdata.cumAngles)[0]
    for b in tdata.bouts.astype(int):
        bb = bias(b[0], b[1], tdata.cumAngles, mca)
        if np.abs(bb) >= strong_thresh:
            safe_set(starting, b[0], 1)
    return starting


def unbiased_bouts(tdata: TailData):
    """
        For a given TailData object returns a bout start array for bouts that are not strongly biased
    """
    if tdata.bouts is None:
        return tdata.starting
    starting = np.zeros(tdata.starting.size, dtype=np.float32)
    mca = mode(tdata.cumAngles)[0]
    for b in tdata.bouts.astype(int):
        bb = bias(b[0], b[1], tdata.cumAngles, mca)
        if -strong_thresh < bb < strong_thresh:
            safe_set(starting, b[0], 1)
    return starting


def left_bouts(tdata: TailData):
    """
    For a given TailData object returns a bout start array for bouts that are likely leftwards
    """
    if tdata.bouts is None:
        return tdata.starting
    starting = np.zeros(tdata.starting.size, dtype=np.float32)
    mca = mode(tdata.cumAngles)[0]
    for b in tdata.bouts.astype(int):
        bb = bias(b[0], b[1], tdata.cumAngles, mca)
        if bb <= -turn_thresh:
            safe_set(starting, b[0], 1)
    return starting


def right_bouts(tdata: TailData):
    """
    For a given TailData object returns a bout start array for bouts that are likely rightwards
    """
    if tdata.bouts is None:
        return tdata.starting
    starting = np.zeros(tdata.starting.size, dtype=np.float32)
    mca = mode(tdata.cumAngles)[0]
    for b in tdata.bouts.astype(int):
        bb = bias(b[0], b[1], tdata.cumAngles, mca)
        if bb >= turn_thresh:
            safe_set(starting, b[0], 1)
    return starting
