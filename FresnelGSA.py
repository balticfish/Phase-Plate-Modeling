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
from Targets import hole, stanford, superTruncGaussian, flatTop, Lenna
import h5py
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import correlate2d
from cmocean import cm as cmo
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from IFTA import IFTA #GS_HIOA
from PlottingTools import StanfordColormap
#from IFTA import IFTA
stanford_colormap = StanfordColormap()

# --- Upgrade the Scipy FFT speed --- 
ncpu = -1
    
def GaussianFit(x, a, omega):
    """
    Gaussian Fit using omega as the 1/e^2 width

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    omega : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return a * np.exp(- (x /omega)**2 /2)

def GaussianWidthFinder(inputBeam, extent):
    """
    

    Parameters
    ----------
    inputBeam : TYPE
        DESCRIPTION.
    extent : TYPE
        DESCRIPTION.

    Returns
    -------
    width_e2 : TYPE
        DESCRIPTION.
    FWHM : TYPE
        DESCRIPTION.

    """
    
    # --- Look at the central horizontal line of the gaussian --- 
    inputIntensity = np.abs(inputBeam)**2
    shape = inputIntensity.shape[0]
    gaussianCut = inputIntensity[:, int(shape/2)]
    xData = np.linspace(extent[0], extent[1], shape)
    pixelSize = 2 * extent[1]/shape
    
    # --- Getting an initial guess from current beam ---
    #maximum amplitude
    aMax = np.max(gaussianCut)
    #width through masking
    widthMask = (gaussianCut > aMax/np.exp(2))
    cutWidth = gaussianCut[widthMask]
    guessWidth = len(cutWidth)* pixelSize/2
    initialGuess = [aMax, guessWidth]

    # --- Optaining the curve fit parameters ---
    popt_gaussian, pcov_gaussian = curve_fit(GaussianFit, xData, gaussianCut, p0=initialGuess)

    # --- Obtaining the width from the curve fit --- 
    yDataFit = GaussianFit(xData, *popt_gaussian)
    
    #For 1/e2
    spline = UnivariateSpline(xData, yDataFit - np.max(yDataFit) / np.exp(2), s=0)
    r1, r2 = spline.roots()
    roots = [r1, r2]
    width_e2 = roots[1] #width as 1/e2 of the intensity
    
    #For FWHM
    splineFWHM = UnivariateSpline(xData, yDataFit - np.max(yDataFit) / 2, s=0)
    FWHM = splineFWHM.roots()[1]
    #print(roots)
    return width_e2, FWHM


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
        Imag = ax1.imshow(field.imag, extent = [extent[0], extent[1], extent[0], extent[1]])
        ax1.set_title("Imaginary Part")
        fig.colorbar(Imag, ax = ax1, orientation = 'horizontal')
        #Real part
        Real = ax2.imshow(field.real, extent = [extent[0], extent[1], extent[0], extent[1]])
        ax2.set_title("Real Part")
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
              extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False,
              gaussianProp = True, save = ''):
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
    plot : Bool, optional
        Boolean to choose if plots should be made. The default is False.
    
    References
    ---------
    I) Fourier Optics and Computational Imaging, Kedar Khare, Chap 11
    
    Returns
    -------
    outputBeam : np.array
        Output Beam in real space with no padding after applying the transfer function
    
    """
    
    # --- Extracting Parameters ---
    k0 = 2 * pi / wavelength
    
    # --- Step 1 : Transforming the input beam to k-space ---
    # Apply the padding to ensure high quality FFT
    paddedBeam = fastOn(inputBeam, padding)
    kBeam = fftshift(fft2(paddedBeam, workers = ncpu))
    kShape = kBeam.shape[0]
    
    
    # --- Step 2 Apply the propagator --- 
    # Creating k-Space coordinates
    kx_ = fftfreq(kShape, d = (2**(1+padding)) * extent[1]/kShape) # 2**(1+padding) * extent[1]/kShape
    ky_ = fftfreq(kShape, d = (2**(1+padding)) * extent[1]/kShape)
    kx_ = fftshift(kx_)
    ky_ = fftshift(ky_)
    kx, ky = np.meshgrid(kx_, ky_)
    

    # Propagator taken from K Khare (see ref)
    propagator = np.exp(1j * z * np.sqrt(k0**2 - 4 * pi**2 * (kx**2 + ky**2)))
    #propagator = np.exp(-1j * z * wavelength * pi *(kx**2 + ky**2))
    
    
    #Apply the propagator in k space
    kPadded = kBeam * propagator
    
    # --- Step 3 : Return to real space ---
    #Return to Cartesian
    outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
    
    #Remove the padding
    outputBeam = fastOff(outputPadded, padding)
    '''plt.imshow(np.abs(outputBeam)**2, cmap = 'jet')
    plt.colorbar()
    plt.show()'''
    if plot:
        inShape = inputBeam.shape[0]
        outputHorizontal, outputVerical  = outputBeam[int(inShape/2), :], outputBeam[:, int(inShape/2)]
        maskPlot = (np.abs(outputVerical)**2 > 5e-6) | (np.abs(outputHorizontal)**2 > 5e-6)
        maskTruth = np.sum(maskPlot)
        
        #outputBeam = outputBeam[int(4*inShape/10):int(6*inShape/10), int(4*inShape/10):int(6*inShape/10)]
        outputBeamReduced = outputBeam[int(inShape/2 - maskTruth):int(inShape/2 + maskTruth),
                                int(inShape/2 - maskTruth):int(inShape/2 + maskTruth)]
        cutIntensityReduced = np.abs(outputBeamReduced[int(outputBeamReduced.shape[1]/2), :])**2
        pixelSize = 2 * extent[1] / inputBeam.shape[0]
        reducedShape = outputBeamReduced.shape[0]
        plotExtent = [-pixelSize * reducedShape/2, pixelSize * reducedShape/2]
        
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.5])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[1,:])
        #Imaginary part
        Imag = ax1.imshow(outputBeamReduced.imag, cmap = cmo.curl,
                          extent = [plotExtent[0], plotExtent[1], plotExtent[0], plotExtent[1]])
        ax1.set_title("Imaginary Part")
        fig.colorbar(Imag, ax = ax1, orientation = 'horizontal')
        #Real part
        Real = ax2.imshow(outputBeamReduced.real, cmap = cmo.curl
                          , extent = [plotExtent[0], plotExtent[1], plotExtent[0], plotExtent[1]])
        ax2.set_title("Real Part")
        fig.colorbar(Real, ax = ax2, orientation = 'horizontal')
        #Intensity
        Intensity = ax3.imshow(np.abs(outputBeamReduced)**2, cmap = stanford_colormap
                               ,extent = [plotExtent[0], plotExtent[1], plotExtent[0], plotExtent[1]])
        ax3.set_title("Intensity")
        fig.colorbar(Intensity, ax = ax3, orientation = 'horizontal')
        #Extra decoration
        # Adding a text box with the width of the beam 1/e^2
        outIntensity = np.abs(outputBeam)**2
        cutIntensity = outIntensity[int(outIntensity.shape[0]/2),:]
        widthMask = (cutIntensity > np.max(outIntensity)/np.exp(2))
        cutWidth = cutIntensity[widthMask]
        outputWidth = len(cutWidth)*pixelSize/2 *1e3
        if gaussianProp:
            outputWidth = GaussianWidthFinder(outputBeam, extent)[0] * 1e3
        textstr = f'Width = {outputWidth:.6g}mm'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        '''ax3.text(0.1, 0.9, textstr, transform=ax3.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)'''

        ax4.axvline(int(reducedShape/2 + 6e-4/pixelSize)+1,linestyle = '--', color = 'k', alpha = 0.5)
        ax4.axvline(int(reducedShape/2 - 6e-4/pixelSize)-1,linestyle = '--', color = 'k', alpha = 0.5)
        
        fig.suptitle("Fresnel Beam after Propagation", size = 20)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax1.set_xlabel('Beam size (m)')
        ax2.set_xlabel('Beam size (m)')
        ax3.set_xlabel('Beam size (m)')
        ax1.set_ylabel('Beam size (m)')
        
        
        #Beam Profile Plot
        ax4.plot(cutIntensityReduced)
        labelSpace = np.linspace(plotExtent[0], plotExtent[1], 7)
        plotSpace = np.linspace(0, plotExtent[1], 7)
        ticks = 2*plotSpace/(pixelSize)
        labels = [np.round(ticker, 4) for ticker in labelSpace]
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-4, 4))
        ax4.xaxis.set_major_formatter(formatter)
        ax4.yaxis.set_major_formatter(formatter)
        ax4.set_xticks(ticks = ticks, labels = labels)
        ax4.set_title("Center cut of the Intensity of the propagated beam")
        ax4.set_xlabel('Beam size (m)')
        ax4.set_ylabel('Intensity')
        fig.tight_layout()
        '''plt.show()
        
        plt.imshow(np.abs(outputBeamReduced)**2, cmap = stanford_colormap, 
                   extent = [plotExtent[0], plotExtent[1], plotExtent[0], plotExtent[1]])
        plt.colorbar()
        plt.xlabel('Distance (m)', size = 13)
        plt.ylabel('Distance (m)', size = 13)
        plt.show()'''
        
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



def Lens(inputBeam, f, wavelength = 253e-9, w0 = 4e-3,
         extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
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
    k0 = 2 * pi / wavelength
    inputShape = inputBeam.shape
    
    # --- Building the transfer function ---
    x_, y_ = np.linspace(extent[0], extent[1], inputShape[0]), np.linspace(extent[0], extent[1], inputShape[1])
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    
    #Built using reference I's radius of curvature implementation
    lensShift = np.exp(-1j * k0 * rSquare/(2 * f))
    
    # --- Applying the transfer function in real space ---
    outputBeam = inputBeam * lensShift
    
    # --- Possible Plotting ---
    if plot:
        plt.imshow(np.abs(outputBeam)**2)
        plt.show()
    
    return outputBeam
    

def phasePlate(inputBeam, hologram = [30, None], wavelength = 253e-9,f = 1.2,
               w0 = 4e-3, extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False,
               realLens = True, boxing = None, rounding = None, SSE = True,
               IFTAPlotting = True, randomSeed = 15, save = ''):
    """
    Phase Plate transfer function

    Parameters
    ----------
    inputBeam : np.array
        The inout beam that should go through the phase plate
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
                       plot = IFTAPlotting, SSE = SSE, rounding = rounding,
                       save = save, realLens = realLens, boxing = boxing, 
                       f = f, z = f, randomSeed = randomSeed)
    else:
        with h5py.File(hologram, 'r') as file:
            hologram = file['Phase'][:]

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
    # --- Globals ---
    wavelength = 253 * 1e-9
    w0 = 8 * 1e-3
    f = 1.2 
    extent = [-1.27 * 1e-2, 1.27 * 1e-2]
    z0 = pi/wavelength * w0**2
    savefile = 'IFTAPhases/TestRun.h5'
    hologramSave = 'IFTAPhases/TestRun.h5'
    randomSeed = 15
    np.random.seed(randomSeed) #Setting the random seed for the IFTA
    
    
    # --- Creating a Phase Plate and Propagating through it --- 
    
    # --- Creating an initial beam to propagate --- 
    inputBeam = Gaussian(sizeFactor=11, plot = True, w0 = w0)
    
    # --- Initializing a target ---
    targetWaist = 10.20116e-4
    target = superTruncGaussian(inputBeam, w0 = targetWaist, trunc = 50)

    # --- Building a phase plate to achieve the given target ---
    plate = phasePlate(inputBeam, plot = True, hologram = [30, target],
                       save = 'Phase8.h5', f = f, randomSeed = randomSeed)# [30,target]
    
    # --- Applying a lens transformation to the beam ---
    lens = Lens(plate, f)
    
    # --- Propagating the beam to the fourier plane --- 
    prop = Propagate(lens, f, plot = True, padding = 0, gaussianProp = False, save = 'SimulatedData8.h5')
    

    
    
    #hologram = 'IFTAPhases/flatTop_round3_2x12_0.6mm_100_1.h5' #the last 2^12 started at w0 = 2e-3
    #hologram = 'IFTAPhases/DataSets/IFTAPhases?latTop_round0_2x11_0.6mm_150.h5'






























