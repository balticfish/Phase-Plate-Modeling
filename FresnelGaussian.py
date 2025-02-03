#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:33:48 2024

Fresnel Optics simulation of a Gaussian beam propagated through a phase plate
and to the fourier plane

Aiming to get the following parameters:
    Wavelength: 253 nm
    Beam size upstream of the phase plate: 8 mm
    Beam size in focus: 1.2 mm
    Target beam profile: cut Gaussian at 50%
    Focusing (Fourier) lens: f=1.2 m


@author: thomas
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import cv2
from numpy import pi as pi
from fft2freq import fft2freq
from numpy import pi
from padding import paddingOn, paddingOff
from gaussianGaussian import gaussian2, gaussianPropagate
# --- Globals --- 
gridSize = 1000
phase = np.random.rand(gridSize, gridSize)

def normalize(array):
    """
    Normalizing a given array 

    Parameters
    ----------
    array : TYPE meshgrid
        meshgrid that should be normalized

    Returns
    -------
    normalized_array : TYPE meshgrid
        

    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def Gaussian(mu_x, mu_y, sigma_x, sigma_y, gridSize = 1000, A0 = 1, phase = True, plot = False):
    """
    Input Gaussian beam simulated from a Fresnel Slow envelope approximation 

    Returns
    -------
    None.

    """
    
    # --- Create a meshgrid --- 
    x = np.linspace(0, gridSize - 1, gridSize)
    y = np.linspace(0, gridSize - 1, gridSize)
    X, Y = np.meshgrid(x, y)
    
    
    # --- Defining the amplitude of the Electric Field ---
    field = A0 * np.exp(-(((X - mu_x)**2 / (2 * sigma_x**2)) +\
                    ((Y - mu_y)**2 / (2 * sigma_y**2))))
        
    # --- Setting the phase --- 
    if not isinstance(phase, np.ndarray):
        phase = np.zeros(gridSize)
    
    # --- Calculate the Gaussian distribution --- 
    gaussian = np.empty_like(field, dtype="complex")
    gaussian.real = field * np.cos(phase)
    gaussian.imag = field * np.sin(phase)
    
    # --- Plotting --- 
    if plot:
        plt.imshow(np.power(np.abs(gaussian.real), 2), cmap = 'jet')
        plt.title('Input Gaussian Beam')
        plt.colorbar()
        plt.show()
    
    return gaussian



def Propagate(inputBeam, z):
    """
    Propagating the beam over a given distance z

    Parameters
    ----------
    input : np.mesgrid -> dtype = complex 
        Input beam to be propagated
    z : distance given in meters
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # --- Extracting globals --- 
    gridSize = inputBeam.shape[0]

    # --- Step 1 transform to k Space --- 
    kBeam = np.fft.fftshift(np.fft.fft2(inputBeam))
    # --- Step 2 Apply the propagator --- 
    
    # --- Creating k space coordinates --- 
    xx_, yy_ = np.meshgrid(np.linspace(0, gridSize - 1, gridSize), 
                         np.linspace(0, gridSize - 1, gridSize), indexing = 'xy')
    kx_, ky_ = fft2freq(xx_, yy_, indexing = 'xy') # From SkUED generalized version
    #kx_, ky_ = np.fft.fftfreq(gridSize, 1), np.fft.fftfreq(gridSize, 1)
    
    kx = np.fft.fftshift(kx_)
    ky = np.fft.fftshift(ky_)
    k = np.hypot(kx, ky)

    propagator = np.empty_like(inputBeam, dtype="complex")
    propagator.imag = -(kx**2 + ky**2)/(2*k) * z
    
    propagator.imag[np.isinf(propagator.imag)] = np.nan
    propagator.imag[np.isnan(propagator.imag)] = np.nanmax(propagator.imag)
    #propagator.imag = normalize(propagator.imag)
    #propagator = np.fft.fftshift(propagator)
    plt.imshow(np.abs(propagator), cmap = 'jet', norm = LogNorm())
    plt.title('Propagator')
    plt.colorbar()
    plt.show()
    
    
    plt.imshow(np.abs(kBeam), norm = LogNorm())
    plt.title('kBeam 1')
    plt.colorbar()
    plt.show()
    
    
    kBeam = kBeam * propagator
    plt.imshow(np.abs(kBeam), norm = LogNorm())
    plt.title('kBeam 2')
    plt.colorbar()
    plt.show()
    
    # --- Step 3 transform back to carthesian space --- 
    
    outputBeam = np.fft.ifft2(np.fft.ifftshift(kBeam))

    return outputBeam 



if __name__ == "__main__":
    """#gaussian = Gaussian(gridSize/2, gridSize/2, gridSize/5, gridSize/5, gridSize = gridSize, plot = True, phase = True)
    gaussian = gaussian2()
    plt.imshow(np.power(np.abs((Propagate(gaussian, 10))), 2), cmap = 'jet')
    plt.colorbar()
    plt.show()"""

   
    beam = gaussian2(w0 = 8e-3)
    plt.imshow(np.abs(beam)**2)
























