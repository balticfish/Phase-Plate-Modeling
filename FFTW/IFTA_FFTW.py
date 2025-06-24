#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:48:11 2024

IFTA Algorithm using FFTW threading

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
#from padding import fastOn, fastOff
def fastOn(inputBeam, size):
    return inputBeam
def fastOff(inputBeam, size):
    return inputBeam
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib.colors import LogNorm
import cv2
from tqdm import tqdm
from Targets import stanford, flatTop, superTruncGaussian
from PhysicalPlate import Box, Density, Density2Level
import h5py
from ToolsIFTA import InitializeLens, InitializePropagator, SSEPlotting
from cmocean import cm
from PlottingTools import StanfordColormap
import gc
import pyfftw
from pyfftw.interfaces.numpy_fft import fftn, ifftn
import multiprocessing
import time
import yaml
pyfftw.interfaces.cache.enable()         # cache plans

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
    
    # Initializing through a smaller phase mask
    #filepath = '/Users/thomas/Desktop/SLAC 2025/Simulations/Phase Plate/IFTAPhases/Manufacturing/Phase_1inch.h5'
    '''runName = 'AWA_FlatTop_2inch_2^12_f=2.5m_target=11.1mm_w0=12e-3_steps=9_iter=300'
    filepath = f'IFTAPhases/CNM/Step_Analysis_2m/Phase_{runName}.h5'
    with h5py.File(filepath, 'r') as file:
        phase = file['Phase'][:]'''
    
    #phase = np.tile([[pi,-pi],[-pi,pi]], (int(inputField.shape[0]/2), int(inputField.shape[1]/2)))
    
    field = np.zeros_like(inputField)
    outputFP = []
    #k0 = 2 * pi / wavelength
    
    # --- Initialising the target amplitude ---
    if target is None:
        raise ValueError('No target was chosen')
        #target = superTruncGaussian(inputField, trunc = 50) #60 #returns an intensity
    target = np.sqrt(target)
    
    
    # --- Initialising the propagation matrices ---
    # This saves on computation time for each iteration
    
    # Initialising the propagator k space for after lens
    propagatorForward = ifftshift(InitializePropagator(inputField, z, padding = size, nProp = nProp,
                                             extent = extent, wavelength = wavelength))
    propagatorBackward = ifftshift(InitializePropagator(inputField, -z, padding = size, nProp = nProp,
                                              extent = extent, wavelength = wavelength))
    # Initialising the propagator k space for before lens (in lens)
    if z0 != None:
        propagatorForward0 = ifftshift(InitializePropagator(inputField, z0, padding = size, nProp = nLens,
                                                  extent = extent, wavelength = wavelength))
        propagatorBackward0 = ifftshift(InitializePropagator(inputField, -z0, padding = size, nProp = nLens,
                                                   extent = extent, wavelength = wavelength))
    # Initialising the lens meshgrid
    lensShiftForward = InitializeLens(inputField, f, nLens = nLens, nProp = nProp,
                                      extent = extent, wavelength = wavelength)
    lensShiftBackward = InitializeLens(inputField, -f, nLens = nLens, nProp = nProp,
                                       extent = extent, wavelength = wavelength)


    # ––– Preparing the FFTWs for the IFTA –––––––––––––––––––––––––––––––––
    # allocate two aligned arrays for in-place FFTW
    a = pyfftw.empty_aligned(propagatorForward.shape, dtype='complex128')
    b = pyfftw.empty_aligned(propagatorForward.shape, dtype='complex128')
    threads = multiprocessing.cpu_count()
    #threads = 64
    print('# of threads used', threads)
    # make plans once:
    t0 = time.perf_counter()
    fft_forward  = pyfftw.FFTW(a, b, axes=(0,1),
                               direction='FFTW_FORWARD',
                               flags=('FFTW_MEASURE',),
                               threads=threads)
    fft_backward = pyfftw.FFTW(b, a, axes=(0,1),
                               direction='FFTW_BACKWARD',
                               flags=('FFTW_MEASURE',),
                               threads=threads)
    t1 = time.perf_counter()
    print(f"FFTW planning took {t1 - t0:.3f} seconds")
    
    # --- Preparing mask for Real Lens ---
    x_, y_ = np.linspace(extent[0], extent[1], inputField.shape[0]), np.linspace(extent[0], extent[1], inputField.shape[1])
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    
    ###################### Adapted to 1/2 inch lens #########################
    outsideLens = (np.sqrt(rSquare) >= 2*1.27e-2) # 1.27e-2/2 for 1/2 inch 
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
    
    for i in tqdm(range(iteration), desc = 'Iterations of IFTA'):
        
        # --- Input the initial Intensity and keep the varying phase --- 
        field = inputAmplitude * np.exp(1j * phase)
    
        # --- Single Lens Optic Propagation ---
        if z0 != None:
            # Propagate the beam to the lens
            a[:] = fastOn(field, size)
            kPadded = fft_forward()
            kPadded *= propagatorForward0
            a[:] = kPadded
            outputPadded = fft_backward()
            field = fastOff(outputPadded, size)
        
        # --- Propagate to Fourier Plane --- 
        # Apply the forward lens shift to the beam
        field *= lensShiftForward
        
        # Propagate the beam to the lens' Focal Plane 
        a[:] = fastOn(field, size)
        kPadded = fft_forward()
        kPadded *= propagatorForward
        a[:] = kPadded
        outputPadded = fft_backward()
        field = fastOff(outputPadded, size)
        
        # --- Extracting Analytics --- 
        if i == int((iteration-1)):
            outputIter = np.power(np.abs(field), 2)
            if save and not SSE:
                outputFP[i, :, :] = outputIter
                saveFile.flush()
        if save and SSE:
            outputFP[i, :, :] = outputIter
            saveFile.flush()
        #outputFP += [outputIter] #output fourier plane intensity at each iteration
        
        
        # --- Replace the intensity with the target intensity --- 
        
        phase = np.angle(field) #Find the new phase after FFT
        field = target * np.exp(1j * phase)
        
        # --- Transform back to Carthesian --- 
        
        # Undoing the propagation
        a[:] = fastOn(field, size)
        kPadded = fft_forward()
        kPadded *= propagatorBackward
        a[:] = kPadded
        outputPadded = fft_backward()
        field = fastOff(outputPadded, size)
        # Undoing the lens transformation
        field *= lensShiftBackward
        # --- Single Lens Optic Propagation ---
        if z0 != None:
            # Propagate the beam to the lens
            a[:] = fastOn(field, size)
            kPadded = fft_forward()
            kPadded *= propagatorBackward0
            a[:] = kPadded
            outputPadded = fft_backward()
            field = fastOff(outputPadded, size)
        
        # --- Collect the new phase --- 
        phase = np.angle(field)
        
        # --- Adjusting the phase to error --- 
        if steps != None: #Sets the phase to discrete levels through rounding error
            phase, thickness = Density(phase, levels = steps, wavelength=wavelength, n_air=nProp, materialIndex=nLens)
            if steps == 3:
                phase, thickness = Density2Level(phase, 0, levels = steps)
        if boxing !=None: #Averages over given pixel size for transverse constraints
            phase = Box(phase, boxing)
        if realLens: #Sets the phase to 0 outside the lens dimensions
            phase[outsideLens] = 0
        gc.collect()
       
       
    if save:
        saveFile.create_dataset('Phase', data=phase)
        saveFile.create_dataset('Target', data=target)
        saveFile.close()
            
        print(f' --- Phase Mask Saved under {save} --- ')

        
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

        """
        # --- Plotting the hologram -> Phase Distribution --- 
        plt.imshow(phase, cmap = cm.curl, extent = [extent[0], extent[1], extent[0], extent[1]])
        plt.title('Post-Phase-Plate Beam Phase')
        plt.xlabel('Distance (m)', fontsize = 14)
        plt.ylabel('Distance (m)', fontsize = 14)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        """
        
        
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

def load_h5_flat(filepath):
    """
    Returns a dict whose keys are:
      • attribute names (from the root group)
      • dataset names (with their full path, e.g. 'inputField', 'extent', etc.)
    and whose values are the attribute or dataset contents.
    """
    cfg = {}
    with h5py.File(filepath, 'r') as f:
        # 1) root attributes
        for name, val in f.attrs.items():
            cfg[name] = val

        # 2) all datasets (at any depth)
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                # read the entire dataset
                cfg[name] = obj[()]
        f.visititems(visitor)

    return cfg

if __name__ == "__main__":
    # ––– Importing the Config through the h5 file ––––––––––––––––––––

    config = load_h5_flat('setupIFTA.h5')
    
    # ––– Launching the IFTA ––––––––––––––––––––––––––––––––––––––––––
    kwargs = {k:v for k,v in config.items() if k not in ('inputField')}
    phase = IFTA(config['inputField'], **kwargs)



