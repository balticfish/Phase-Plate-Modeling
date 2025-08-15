#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:11:54 2024

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

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from padding import fastOn, fastOff
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from Targets import hole, stanford, superTruncGaussian, flatTop, Lenna, Aligner, Aligner2, CathodeCleaner, LaserHeater2
import h5py
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import correlate2d
from cmocean import cm as cmo
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from IFTA import IFTA #GS_HIOA
from PlottingTools import StanfordColormap, plotBeam, AutoCorr
from HermiteGaussianFit import hermiteGaussianModes, fullHGField, HermiteGaussianMode, Plotting
from scipy.ndimage import gaussian_filter
from PhysicalPlate import Density
from IFTAZone import IFTAZone
import RefractionIndex

stanford_colormap = StanfordColormap()

# --- Upgrade the Scipy FFT speed --- 
ncpu = -1
    



def Gaussian(sizeFactor = 11, wavelength = 253e-9, w0 = 4e-3,
             extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    """
    Creates a Gaussian Beam for use in the Fresnel Optics Propagator

    Parameters
    ----------
    sizeFactor : int, optional
        The power that the size should be raised such that shape = 2^size +1. The default is 11.
    wavelength : float, optional
        Single wavelength of the beam. The default is 253 nm
    w0 : float, optional
        The waist of the curve using 1/e of the amplitude. The default is 4 mm
    extent : array, optional
        Extent of the array to build. The default is set in the globals.
    plot : Bool, optional
        Boolean to choose if plots should be made. The default is False.

    Returns
    -------
    field : np.array
        The complex electric field of the input gaussian beam

    """
    # --- Extracting key paramters ---
    z0 = pi/wavelength * w0**2
    q0 = 1j * z0
    k0 = 2* pi / wavelength
    
    # --- Build a meshgrid to apply the gaussian too ---
    gridSize = 2 ** sizeFactor +1 # if 2n+1 add here
    x_ = np.linspace(extent[0], extent[1], gridSize)
    y_ = np.linspace(extent[0], extent[1], gridSize)
    X, Y = np.meshgrid(x_, y_)
    rSquare = X**2 + Y**2
    
    # --- Creating the gaussian field using complex beam parameter ---    
    field = 1/q0 * np.exp(- 1j * k0 * rSquare / (2 * q0))
    
    
    # --- Plotting the field if required --- 
    if plot:
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        #Imaginary part
        Imag = ax1.imshow(np.angle(field), extent = [extent[0], extent[1], extent[0], extent[1]], cmap = stanford_colormap)
        ax1.set_title("Phase")
        fig.colorbar(Imag, ax = ax1, orientation = 'horizontal')
        #Real part
        Real = ax2.imshow(np.abs(field), extent = [extent[0], extent[1], extent[0], extent[1]], cmap = stanford_colormap)
        ax2.set_title("Amplitude")
        fig.colorbar(Real, ax = ax2, orientation = 'horizontal')
        #Intensity
        Intensity = ax3.imshow(np.abs(field)**2, cmap = stanford_colormap, 
                               extent = [extent[0], extent[1], extent[0], extent[1]])
        ax3.set_title("Intensity")
        fig.colorbar(Intensity, ax = ax3, orientation = 'horizontal')
        #Extra decoration
        fig.suptitle("Input Gaussian Beam Parts", size = 20)
        ax1.set_xlabel('Beam size (m)')
        ax2.set_xlabel('Beam size (m)')
        ax3.set_xlabel('Beam size (m)')
        ax1.set_ylabel('Beam size (m)')
        fig.tight_layout()
        plt.show()
    
    return field


def Propagate(inputBeam, z, wavelength = 253e-9, w0 = 4e-3, padding = 1,
              extent = [-1.27 * 1e-2, 1.27 * 1e-2], save = '', n= 1.00030067):
    """
    Applying the propagator transfer function to an input Complex Beam 

    Parameters
    ----------
    inputBeam : np.array
        The complex beam to apply the transfer function to
    z : float
        The distance to propagate by
    wavelength : float, optional
        Single wavelength of the beam. The default is 253 nm
    w0 : float, optional
        The waist of the curve using 1/e of the amplitude. The default is 4 mm
    padding : integer, optional
        factor of 2 to pad the array with -> See padding.py. The default is 1.
    extent : array, optional
        Extent of the array to build. The default is set in the globals.
    
    References
    ---------
    I) Fourier Optics and Computational Imaging, Kedar Khare, Chap 11
    
    Returns
    -------
    outputBeam : np.array
        Output Beam in real space with no padding after applying the transfer function
    
    """
    
    # --- Extracting Parameters ---
    k0 = 2 * pi / (wavelength/n)
    
    # --- Step 1 : Transforming the input beam to k-space ---
    # Apply the padding to ensure high quality FFT
    paddedBeam = fastOn(inputBeam, padding)
    kBeam = fftshift(fft2(paddedBeam, workers = ncpu))
    kShape = kBeam.shape[0]
    
    
    # --- Step 2 Apply the propagator --- 
    # Creating k-Space coordinates
    kx_ = 2*pi*fftfreq(kShape, d = (2**(1+padding)) * extent[1]/kShape) 
    ky_ = 2*pi*fftfreq(kShape, d = (2**(1+padding)) * extent[1]/kShape)
    kx_ = fftshift(kx_)
    ky_ = fftshift(ky_)
    kx, ky = np.meshgrid(kx_, ky_)
    
    
    # Propagator taken from K Khare (see ref)
    propagator = np.exp(1j * z * np.sqrt(k0**2 -  (kx**2 + ky**2)))#4 * pi**2 *
    #propagator = np.exp(-1j * z * wavelength * pi *(kx**2 + ky**2))
    
    
    #Apply the propagator in k space
    kPadded = kBeam * propagator

    # --- Step 3 : Return to real space ---
    #Return to Cartesian
    outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
    
    #Remove the padding
    outputBeam = fastOff(outputPadded, padding)
        
        
    if save:
        inShape = inputBeam.shape[0]
        outputHorizontal, outputVertical  = outputBeam[int(inShape/2), :], outputBeam[:, int(inShape/2)]
        pixelSize = 2 * extent[1] / inputBeam.shape[0]
        print(' --- Fresnel Propagated Data Saved under:', save, '---')
        with h5py.File(save, 'w') as file:
            file.create_dataset('OutputBeam', data = outputBeam)
            file.create_dataset('IntensityVertical', data=outputVertical)
            file.create_dataset('IntensityHorizontal', data = outputHorizontal)    
            file.create_dataset('ExtentTotal', data = extent)
            file.create_dataset('PixelSize', data = pixelSize)
            file.create_dataset('InputShape', data = inputBeam.shape)
        
    return outputBeam 



def Lens(inputBeam, f, wavelength = 253e-9, w0 = 4e-3, nLens = 1.5058500198911624,
         nProp = 1.00030067, extent = [-1.27 * 1e-2, 1.27 * 1e-2]):
    """
    Applying a lens transformation to an incoming Complex Beam

    Parameters
    ----------
    inputBeam : np.array
        The complex beam to apply the transfer function to
    f : float
        The focal length of the lens to use (assumed symmetrical in x and y)
    wavelength : float, optional
        Single wavelength of the beam. The default is 253 nm
    w0 : float, optional
        The waist of the curve using 1/e of the amplitude. The default is 4 mm
    extent : array, optional
        Extent of the array to build. The default is set in the globals.
    plot : Bool, optional
        Boolean to choose if plots should be made. The default is False.
        
    References
    ----------
    I) 'Soft x-ray self-seeding simulation methods and their application for 
        the Linac Coherent Light Source', S. Serkez et al.

    Returns
    -------
    outputBeam : np.array
        Outgoing Complex Beam after Lens  transformation

    """
    
    # --- Extracting Parameters --- 
    k0 = 2 * pi / (wavelength) # How do I adapt the wavelength for inside the lens ? 
    
    # --- Adapting the focal length to the propagation medium ---
    if nLens != None:
        f = f * (1-nLens)/(nProp - nLens)
        #k0 *= nLens#nLens
    
    inputShape = inputBeam.shape
    
    # --- Building the transfer function ---
    x_, y_ = np.linspace(extent[0], extent[1], inputShape[0]), np.linspace(extent[0], extent[1], inputShape[1])
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    
    
    #Built using reference I's radius of curvature implementation
    lensShift = np.exp(-1j * k0 * rSquare/(2 * f))
    
    # --- Applying the transfer function in real space ---
    outputBeam = inputBeam * lensShift
    
    return outputBeam
    

def phasePlate(inputBeam, hologram = [30, None], wavelength = 253e-9, f = 1.2,
               w0 = 4e-3, extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False,
               realLens = True, boxing = None, steps = None, SSE = True, z0 = None,
               IFTAPlotting = True, randomSeed = 15, save = '', nProp = 1, nLens = 1,
               size = 0):
    """
    Phase Plate transfer function
    
    Parameters
    ----------
    inputBeam : np.array
        The input beam that should go through the phase plate
    hologram : array or string, optional
        if array -> [iterations of GSA, GSA target], 
        This will launch an IFTA phase retrieval for the target The default is [30, None].
        if string -> This is the save file of type 'filename.h5'
    wavelength : float, optional
        Single wavelength of the beam. The default is 253 nm
    w0 : float, optional
        The waist of the curve using 1/e of the amplitude. The default is 4 mm
    extent : array, optional
        Extent of the array to build. The default is set in the globals.
    plot : Bool, optional
        Boolean to choose if plots should be made. The default is False.

    Returns
    -------
    outputBeam : np.array
        meshgrid of the beam after passing through the phase plate

    """    
    
    if len(hologram) == 2:
        iterations, target = hologram
        hologram = IFTA(inputBeam, iteration = iterations, target = target,
                       plot = IFTAPlotting, SSE = SSE, steps = steps,
                       save = save, realLens = realLens, boxing = boxing, z0 = z0,
                       f = f, z = f, randomSeed = randomSeed, nProp = nProp, nLens = nLens,
                       extent = extent, size = size)
    else:
        with h5py.File(hologram, 'r') as file:
            hologram = file['Phase'][:]
    
    print('Unique phase levels in phase plate', np.unique(hologram))
    
    # --- Adding etching uncertainty ---
    '''uncertainty = np.zeros(hologram.shape)
    #print(np.unique(hologram))
    randomOffset = np.random.rand(len(np.unique(hologram)))

    for i, phase in enumerate(np.unique(hologram)):

        uncertainty[hologram == phase] = randomOffset[i] * 0.011175626040438 * 0
        
    uncertainty = MaskOptics(uncertainty, 1.27e-2, extent = extent)
    hologram += uncertainty'''
    inputPhase = np.angle(inputBeam)
    phasePlate = np.subtract(hologram, inputPhase)
    
    outputBeam = inputBeam * np.exp(1j * phasePlate)
    
    
    if plot:
        fig, (axA, axB) = plt.subplots(2, 2, figsize=(12, 10))
        
        outphase = np.angle(outputBeam)
        subtract = axA[0].imshow(outphase, cmap = 'vlag', extent = [extent[0], extent[1], extent[0], extent[1]])
        Real = axA[1].imshow(outputBeam.real, extent = [extent[0], extent[1], extent[0], extent[1]])
        Imag = axB[0].imshow(outputBeam.imag, extent = [extent[0], extent[1], extent[0], extent[1]])
        Intensity = axB[1].imshow(np.abs(outputBeam)**2, cmap = stanford_colormap
                                  , extent = [extent[0], extent[1], extent[0], extent[1]])
        
        fig.colorbar(subtract, ax = axA[0], orientation = 'vertical')
        fig.colorbar(Real, ax = axA[1], orientation = 'vertical')
        fig.colorbar(Imag, ax = axB[0], orientation = 'vertical')
        fig.colorbar(Intensity, ax = axB[1], orientation = 'vertical')

        axA[0].set_title("Output Phase")
        axA[1].set_title("Real Part")
        axB[0].set_title("Imaginary part")
        axB[1].set_title("Intensity")
        
        axA[0].set_xlabel('Beam size (m)')
        axA[1].set_xlabel('Beam size (m)')
        axB[0].set_xlabel('Beam size (m)')
        axB[1].set_xlabel('Beam size (m)')
        axA[0].set_ylabel('Beam size (m)')
        fig.suptitle("Beam Output Characterstics Post-Phase-Plate", size = 20)
        fig.tight_layout()
        plt.show()
        
    return outputBeam


def MaskOptics(inputBeam, maskRadius, extent = [1.27e-2, 1.27e-2]):
    """
    Apply a crude Iris to the beam removing the field outside of the given aperture

    Parameters
    ----------
    inputBeam : np.array
        complex field array containing the beam
    maskRadius : float
        Radius of the aperture to be applied
    extent : list, optional
        Given extent of the array in m. The default is [1.27e-2, 1.27e-2].

    Returns
    -------
    outputBeam : np.array
        complex field array after the aperture is applied

    """
    # --- Extract Data --- 
    inputShape = inputBeam.shape
    outputBeam = np.array(inputBeam) # Create a copy of the beam
    
    # --- Creating a grid to set the aperture to --- 
    x_, y_ = np.linspace(extent[0], extent[1], inputShape[0]), np.linspace(extent[0], extent[1], inputShape[1])
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    
    # --- Setting the field to 0 outside of the aperture --- 
    outputBeam[(np.sqrt(rSquare) > maskRadius)] = 0
    
    return outputBeam


if __name__ == "__main__":
    # ––– Globals –––
    wavelength = 253e-9#257.5 * 1e-9
    w0 = 8*1e-3    # Beam waist radius in 1/e^2
    f = 1.2#1.2968270716706576
    extentFactor = 1
    extent = extentFactor * np.array([-1.27 * 1e-2, 1.27 * 1e-2])
    z0 = pi/wavelength * w0**2
    randomSeed = 20#21  # 30 was used before zone plate testing
    np.random.seed(randomSeed)    #Setting the random seed for the IFTA
    nLens = RefractionIndex.FusedSilica(wavelength)    #1.5058500198911624 # None
    nProp = RefractionIndex.Air(wavelength)    #1.00030067 # 1

    z1 = (2.2e-3) / 2    # None #Half thickness of a 1m lens
    sizeFactor = 11
    targetRadius = (1.2/2)*1e-3    # Radius of the target that should be achieved after propagation
    trunc = 50#63#69.8 #50    # Truncation Percentage # 63
    iterations = 100
    steps = 3
    
    # ––– Save Files –––
    savePhase = 'Phase.h5'
    saveData = 'SimulatedData.h5'

    
    ###########################################################
    # --- Creating a Phase Plate and Propagating through it --- 
    ###########################################################
    
    # ––– Creating a Pure Gaussian Beam to Propagate ––––––––––––––––––––––––––
    inputBeam = Gaussian(sizeFactor = sizeFactor,
                         plot = True, w0 = w0, extent = extent)
    inputBeam = MaskOptics(inputBeam, extent[1], extent = extent) #Cutting the input beam at 1 inch aperture
    
    # ––– Initializing a target –––––––––––––––––––––––––––––––––––––––––––––––
    target = superTruncGaussian(inputBeam, targetRadius = targetRadius,
                                trunc = trunc, extent = extent)
    
    
    # ––– Building a phase plate to achieve the given target ––––––––––––––––––
    print("\n Starting IFTA ...\n")
    plate = phasePlate(inputBeam, plot = True, nProp = nProp, nLens = nLens, hologram = NanoX,
                       save = '', f = f, z0 = z1, randomSeed = randomSeed, steps = steps, extent = extent,
                       size = 0)# [iterations, target] / savePhase
    
    np.unique(plate)

    
    # --- Applying the diffraction between the phase mask and the lens ---
    midLens = Propagate(plate, z1, padding = 0, n = nLens, save = '')
    
    # --- Applying a lens transformation to the beam ---
    lens = Lens(midLens, f, nLens = nLens, nProp = nProp, extent = extent)
    
    prop = Propagate(lens, f, padding = 0, n = nProp, save = saveData, extent = extent) #saveData
    
    plotBeam(prop, extent = extent, truncRadius = targetRadius, fitCut = True, title = 'Beam Profile at Cathode', maxROI = 60)
    
    
    AutoCorr(prop, extent = extent, truncRadius = targetRadius, title = "Characteristic Speckle", clip = 0.5)
    
