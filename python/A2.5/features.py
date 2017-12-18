# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This file is used for extracting features over windows of tri-axial accelerometer
data. We recommend using helper functions like _compute_mean_features(window) to
extract individual features.

As a side note, the underscore at the beginning of a function is a Python
convention indicating that the function has private access (although in reality
it is still publicly accessible).

"""


import numpy as np
import math

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window, axis=0)

def _compute_variance_feature(window):
    return  np.var(window,axis=0)

def _compute_zerocrossingrate_feature(window):
    return np.sum(np.count_nonzero(np.diff(np.sign(window),axis=0),axis=0))


def _compute_magnitudesignal(window):
    return np.mean(np.linalg.norm(window,axis=1))


def _compute_FFT(window):
    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]
    frep = np.fft.rfftfreq(window.size/3)
    xfft = np.fft.rfft(x, n=window.size/3).astype(float)
    yfft = np.fft.rfft(y, n=window.size/3).astype(float)
    zfft = np.fft.rfft(z, n=window.size/3).astype(float)
    return np.array([[frep[xfft.argmax()]], [frep[yfft.argmax()]], [frep[zfft.argmax()]]])

def _computer_entropy(window):
    hist1 = np.histogram(window,bins=5)[0]
    hist2 = [p for p in hist1 if p != 0]
    entropy = [ -p * math.log(abs(p)) for p in hist2]
    return np.sum(entropy)

def _compute_acceleration(window):
    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]

    Ix = _distance(x[0],x[1])
    Sx = 0.0
    for i in range(window.size/3-1):
        Sx = Sx + _distance(x[i],x[i+1])
    Ax = 2*(Sx - Ix*(window.size/3-1))/(window.size/3-1)**2

    Iy = _distance(y[0],y[1])
    Sy = 0.0
    for i in range(window.size/3-1):
        Sy = Sy + _distance(y[i],y[i+1])
    Ay = 2*(Sy - Iy*(window.size/3-1))/(window.size/3-1)**2

    Iz = _distance(z[0],z[1])
    Sz = 0.0
    for i in range(window.size/3-1):
        Sz = Sz + _distance(z[i],z[i+1])
    Az = 2*(Sz - Iz*(window.size/3-1))/(window.size/3-1)**2

    return [Ax,Ay,Az]

def _distance(x1,x2):
    return math.sqrt((x2-x1)**2)

def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature matrix X.

    Make sure that X is an N x d matrix, where N is the number
    of data points and d is the number of features.

    """

    x = []
    x = np.append(x, _compute_mean_features(window))
    x = np.append(x, _compute_variance_feature(window))
    x = np.append(x, _compute_zerocrossingrate_feature(window))
    x = np.append(x, _compute_magnitudesignal(window))
    x = np.append(x, _compute_FFT(window))
    x = np.append(x, _computer_entropy(window))
    x = np.append(x, _compute_acceleration(window))
    return x
