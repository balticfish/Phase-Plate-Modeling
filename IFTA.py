#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:48:11 2024

IFTA Algorithm using propagation

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from padding import fastOn, fastOff
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib.colors import LogNorm
import cv2
from tqdm import tqdm
from Targets import stanford, flatTop, superTruncGaussian
from PhysicalPlate import Box, Density, Density2Level
import h5py
from ToolsIFTA import InitializeLens, InitializePropagator, SSEPlotting, Lens
from cmocean import cm
from PlottingTools import StanfordColormap
import gc
import RefractionIndex
stanford_colormap = StanfordColormap()

# --- Globals ---
# --- Upgrade the Scipy FFT speed --- 
ncpu = -1



def IFTA(inputField, iteration = 30, f = 1.2, z = 1.2, target = None,
        size = 0, plot = False, SSE = False, save = '', nProp = 1.00030067,
        steps = None, boxing = None, wavelength = 253e-9, randomSeed = 15, z0 = 1.1e-3,
        nLens = 1.5058500198911624, extent = [-1.27 * 1e-2, 1.27 * 1e-2], realLens = None):
    """
    
    Optical Iterative Fourier Transform Algorithm 
    Produces Phase Masks for Phase Plate Simulations

    Parameters
    ----------
    inputField : np.array
        The input beam's complex field 
    iteration : int, optional
        The number of iterations for the IFTA. The default is 30.
    target : np.array, optional
        Target Array that wants to be achieved after focusing. 
        The default is None which creates a 50% truncated Guassian
    size : int, optional
        Creates additional padding for more precise Fourier Transforms. 
        The int sets the array to the next fast power of 2
        The default is 0 which keeps the array the same size
    plot : Bool, optional
        Plots the outputs. The default is False.
    SSE: Bool, optional
        Performs an Normalized Sum Squared Error Analysis on the data
        as a goodness of fit test. The default is False.
    save : string, optional
        filepath location for saving. The default is '' where the file will not be saved.

    Returns
    -------
    phase : np.array
        Outputs the phase mask used for the phase plate

    """
    # --- Intialising parameters --- 
    inputAmplitude = np.abs(inputField)
    phase = np.angle(np.array(inputField))
    
    # --- Initializes a randomized phase mask to improve the gradient descent --- 
    np.random.seed(randomSeed) #Setting a randomized seed for 
    phase = np.random.rand(inputField.shape[0], inputField.shape[1]) * pi
    
    #phase = (np.max(inputAmplitude) - inputAmplitude)*40*pi
    #nLens     = RefractionIndex.FusedSilica(wavelength)
    #nProp     = RefractionIndex.Air(wavelength)
    '''phase     = np.angle(Lens(inputField, -6.85, nLens = nLens, nProp = nProp, extent = extent)) #-4.3388#-6.784
    phase    += (np.random.rand(inputField.shape[0], inputField.shape[1]) - 0.5) *2* pi/2'''

    #phase += np.random.randint(2, size = (inputField.shape[0], inputField.shape[1])) * pi/24
    #phase = phase%(2*pi)

    # Initializing through a smaller phase mask
    '''filepath = 'Phase.h5'
    with h5py.File(filepath, 'r') as file:
        phase = file['Phase'][:]
        phase += np.random.rand(inputField.shape[0], inputField.shape[1]) * 1e-7
    '''
    
    #phase = np.tile([[pi,-pi],[-pi,pi]], (int(inputField.shape[0]/2), int(inputField.shape[1]/2)))
    
    field = np.zeros_like(inputField)
    outputFP = []
    #k0 = 2 * pi / wavelength
    
    # --- Initialising the target amplitude ---
    if target is None:
        target = superTruncGaussian(inputField, trunc = 50) #60 #returns an intensity
    target = np.sqrt(target)
    
    
    # --- Initialising the propagation matrices ---
    # This saves on computation time for each iteration
    
    # Initialising the propagator k space for after lens
    propagatorForward = InitializePropagator(inputField, z, padding = size, nProp = nProp,
                                             extent = extent, wavelength = wavelength)
    propagatorBackward = InitializePropagator(inputField, -z, padding = size, nProp = nProp,
                                              extent = extent, wavelength = wavelength)
    # Initialising the propagator k space for before lens (in lens)
    if z0 != None:
        propagatorForward0 = InitializePropagator(inputField, z0, padding = size, nProp = nLens,
                                                  extent = extent, wavelength = wavelength)
        propagatorBackward0 = InitializePropagator(inputField, -z0, padding = size, nProp = nLens,
                                                   extent = extent, wavelength = wavelength)
    # Initialising the lens meshgrid
    lensShiftForward = InitializeLens(inputField, f, nLens = nLens, nProp = nProp,
                                      extent = extent, wavelength = wavelength)
    lensShiftBackward = InitializeLens(inputField, -f, nLens = nLens, nProp = nProp,
                                       extent = extent, wavelength = wavelength)
    
    # --- Preparing mask for Real Lens ---
    x_, y_ = np.linspace(extent[0], extent[1], inputField.shape[0]), np.linspace(extent[0], extent[1], inputField.shape[1])
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    
    ###################### Adapted to 1/2 inch lens #########################
    outsideLens = (np.sqrt(rSquare) >= 1.27e-2) # 1.27e-2/2 for 1/2 inch 
    #########################################################################
    
    
    # --- Creating the save file ---
    
    try:
        # if hf exists and is still open...
        if isinstance(hf, h5py.File) and hf.id.valid:
            hf.close()
    except NameError:
        # hf was never created, so nothing to close
        pass

    if save:
        saveFile = h5py.File(save, 'w')
        outputFP = saveFile.create_dataset(
            'OutputFP',
            shape = (iteration, inputField.shape[0], inputField.shape[1]),
            chunks = (1, inputField.shape[0], inputField.shape[1])
            #dtype = 'float64'
            )
    else:
        outputFP = None

    # ––– Determine the bandpass filter mask for the padded beam –––
    # To remove high freq noise due to the FFT
    kx_ = 2*pi*fftfreq(propagatorForward.shape[0], d = (2**(1+padding)) * extent[1]/inputAmplitude.shape[0]) 
    ky_ = 2*pi*fftfreq(propagatorForward.shape[1], d = (2**(1+padding)) * extent[1]/inputAmplitude.shape[1])
    #kx_ = fftshift(kx_)
    #ky_ = fftshift(ky_)
    kx, ky = np.meshgrid(kx_, ky_)
    Ks  = np.sqrt(kx**2 + ky**2)
    bandpass = (Ks>k_max)
    
    for i in tqdm(range(iteration), desc = 'Iterations of IFTA'):
        
        # --- Input the initial Intensity and keep the varying phase --- 
        field = inputAmplitude * np.exp(1j * phase)
    
        # --- Single Lens Optic Propagation ---
        if z0 != None:
            # Propagate the beam to the lens
            paddedField = fastOn(field, size)
            kPadded = fftshift(fft2(paddedField, workers = ncpu))
            kPadded *= propagatorForward0
            kPadded[bandpass] = 0+1j*0
            outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
            field = fastOff(outputPadded, size)
        
        # --- Propagate to Fourier Plane --- 
        # Apply the forward lens shift to the beam
        field *= lensShiftForward
        
        # Propagate the beam to the lens' Focal Plane 
        paddedField = fastOn(field, size)
        kPadded = fftshift(fft2(paddedField, workers = ncpu))
        kPadded *= propagatorForward
        kPadded[bandpass] = 0+1j*0
        outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
        field = fastOff(outputPadded, size)
        
        # --- Extracting Analytics --- 
        outputIter = np.power(np.abs(field), 2)
        if save:
            outputFP[i, :, :] = outputIter
            saveFile.flush()
        #outputFP += [outputIter] #output fourier plane intensity at each iteration
        
        
        # --- Replace the intensity with the target intensity --- 
        
        phase = np.angle(field) #Find the new phase after FFT
        field = target * np.exp(1j * phase)
        
        # --- Transform back to Carthesian --- 
        
        # Undoing the propagation
        paddedField = fastOn(field, size)
        kPadded = fftshift(fft2(paddedField, workers = ncpu))
        kPadded *= propagatorBackward
        kPadded[bandpass] = 0+1j*0
        outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
        field = fastOff(outputPadded, size)
        # Undoing the lens transformation
        field *= lensShiftBackward
        # --- Single Lens Optic Propagation ---
        if z0 != None:
            # Propagate the beam to the lens
            paddedField = fastOn(field, size)
            kPadded = fftshift(fft2(paddedField, workers = ncpu))
            kPadded *= propagatorBackward0
            kPadded[bandpass] = 0+1j*0
            outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
            field = fastOff(outputPadded, size)
        
        # --- Collect the new phase --- 
        phase = np.angle(field)
        
        # --- Adjusting the phase to error --- 
        if steps != None: #Sets the phase to discrete levels through rounding error
            if steps == 3:
                phase, thickness = Density2Level(phase, 0, levels = steps, wavelength=wavelength, n_air=nProp, materialIndex=nLens)

                #phase[(phase==pi)] = 9*pi/10
            else:
                phase, thickness = Density(phase, levels = steps, wavelength=wavelength, n_air=nProp, materialIndex=nLens)
                
            phase[(phase==-pi)] = pi
            
        if boxing !=None: #Averages over given pixel size for transverse constraints
            phase = Box(phase, boxing)
        if realLens != None: #Sets the phase to 0 outside the lens dimensions
            phase[outsideLens] = 0
        gc.collect()
       
    # --- If plot is set to True the outputs will be neatly plotted ---
    if plot:
        # --- Plotting the Fourier Plane Intensity --- 
        plt.imshow(outputIter, cmap = stanford_colormap, extent = [extent[0], extent[1], extent[0], extent[1]])#, norm = LogNorm())
        plt.title('Fourier Plane Output Intensity')
        plt.xlabel('distance (m)')
        plt.ylabel('distance (m)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
        # --- Plotting the hologram -> Phase Distribution --- 
        plt.imshow(phase, cmap = cm.curl, extent = [extent[0], extent[1], extent[0], extent[1]])
        plt.title('Post-Phase-Plate Beam Phase')
        plt.xlabel('Distance (m)', fontsize = 14)
        plt.ylabel('Distance (m)', fontsize = 14)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    
       
    if save:
        saveFile.create_dataset('Phase', data=phase)
        saveFile.create_dataset('Target', data=target)
        saveFile.close()
            
        print(f' --- Phase Mask Saved under {save} --- ')
    
        
    # --- If set the SSE will be computed and plotted if plot is asked --- 
    if SSE:
        # --- Plotting the quality of the output throughout the process --- 
        if save:
            with h5py.File(save, 'r') as file:
                fpStack = file['OutputFP'][...]    
            SSE_result = SSEPlotting(fpStack, target, plotting = True)
            del fpStack
            gc.collect()
            with h5py.File(save, 'a') as file:
                file.create_dataset('SSE', data=SSE_result)
                
    return phase



if __name__ == "__main__":
    # ––– Test Case –––
    
    # ––– Globals –––
    extent = [-1.5 * 1e-2, 1.5 * 1e-2]
    wavelength = 253 * 1e-9
    w0 = 6 * 1e-3
    f = 1.2

    # ––– Testing the IFTA –––
    z0 = pi/wavelength * w0**2
    q0 = 1j * z0
    k0 = 2* pi / wavelength
    
    # ––– Build a meshgrid to apply the gaussian too –––
    gridSize = 2 ** 11 +1
    x_ = np.linspace(extent[0], extent[1], gridSize)
    y_ = np.linspace(extent[0], extent[1], gridSize)
    X, Y = np.meshgrid(x_, y_)
    rSquare = X**2 + Y**2
    
    # ––– Creating the gaussian field using complex beam parameter –––    
    field = 1/q0 * np.exp(- 1j * k0 * rSquare / (2 * q0))
    plt.imshow(np.abs(field)**2)
    plt.show()
    
    # ––– Initializing a target –––
    target = flatTop(field, extent = extent, w0 = 6e-4, plot = True)
    
    # ––– Finding the phase necessary for transformation –––
    phase = IFTA(field, plot = True, SSE=True, iteration = 30, extent = extent,
                 target = None, size = 1, boxing = None, realLens = True, steps = None)
    
