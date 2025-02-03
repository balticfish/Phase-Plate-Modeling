#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:24:36 2024

Tool functions for the IFTA algorithms

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
import pandas as pd
from PhysicalPlate import Box
import h5py
from PlottingTools import StanfordColormap

# --- Initializing Globals --- 
stanford_colormap = StanfordColormap()
ncpu = -1

def normalize(array):
    """
    Normalizing a given array 

    Parameters
    ----------
    array : meshgrid
        meshgrid that should be normalized

    Returns
    -------
    normalized_array : meshgrid
        normalized meshgrid 

    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array



def hologram(phase):
    """
    Sets the range value for a given array between -2pi and 2pi
    
    Parameters
    ----------
    phase : array
        phase array from the ITFA

    Returns
    -------
    holo : array
        hologram 

    """
    phase = np.where(phase<0, phase+2*np.pi, phase)
    p_max = np.max(phase)
    p_min = np.min(phase)
    holo = pi * ((phase - p_min)/(p_max- p_min))
    return holo

def SSE(field1, field2):
    """
    Compute the Sum Squared Error of two arrays 

    Parameters
    ----------
    field1 : array
        array representing the first beam
    field2 : array
        array representing the second beam

    Returns
    -------
    SSE : float
        Normalized Summed Squared Error between the two arrays

    """
    field2 = np.array(normalize(field2), dtype = np.float64)
    field1 = np.array(normalize(field1), dtype = np.float64)
    # --- Take the difference between both amplitudes --- 
    subtract = np.subtract(field1, field2)
    # --- Get the Summed Squared Error ---
    squared = np.square(subtract)
    summed1 = np.sum(squared.flatten())
    # --- Get the Summed Squared Target --- 
    squared2 = np.square(field2)
    summed2 = np.sum(squared2.flatten())
    return 10 * np.log10(summed1/summed2)

def SSEPlotting(outputs, target, plotting = True):
    """
    Plotting the Normalized Summed Squared Error between the target and IFTA outputs

    Parameters
    ----------
    outputs : array
        array of propagated beams after the phase plate at the fourier plane
    target : array
        array representing the target to compare data with
    plotting : Bool, optional
        When True the fucntion will plot the outputs. The default is True.

    Returns
    -------
    quality : array
        result of the the scalar product betweeen each output and the target

    """
    quality = []
    for field in tqdm(range(len(outputs)), desc = 'SSE IFTA Verification'):
        quality += [SSE(outputs[field], target)]
    if plotting:
        plt.plot(np.arange(1, len(quality)+1),quality, '.-')
        plt.title('Quality check from Normalized Summed Square Error')
        plt.xlabel("Iterations")
        plt.ylabel("SSE in dB")
        plt.xscale('log', base = 10)
        maxQuality = quality[-1]
        textstr = f'Final Quality = {maxQuality:.4g} dB'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.4, 0.9, textstr,  fontsize=14,transform=plt.gca().transAxes,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()
    return quality

def scalarProduct(field1, field2):
    """
    Compute the scalar product of two electric fields.

    Parameters:
    ----------
    field1 (ndarray): First electric field (complex-valued).
    field2 (ndarray): Second electric field (complex-valued).

    Returns:
    ----------
    float: The scalar product of the two fields.
    """
    # --- Flatten the fields to 1D arrays ---
    field1_flat = field1.flatten()
    field2_flat = field2.flatten()
    
    # --- Compute the scalar product --- 
    scalar_product_value = np.sum(field1_flat * np.conjugate(field2_flat))
    
    # --- Normalize by the product of norms to get the overlap --- 
    norm1 = np.sqrt(np.sum(np.abs(field1_flat)**2))
    norm2 = np.sqrt(np.sum(np.abs(field2_flat)**2))
    
    if norm1 != 0 and norm2 != 0:
        normalized_scalar_product = scalar_product_value / (norm1 * norm2)
    else:
        normalized_scalar_product = 0
    
    return np.abs(normalized_scalar_product)

def scalarPlotting (outputs, target, plotting = True):
    """
    Plotting the scalar product through each iteration of IFTA

    Parameters
    ----------
    target : array
        array representing the target to compare data with
    outputs : array
        array of propagated beams after the phase plate at the fourier plane

    Returns
    -------
    quality : array
        result of the the scalar product betweeen each output and the target
    
    References
    -------
    Direct fabrication of arbitrary phase masks in optical glass via 
    ultra-short pulsed laser writing of refractive index modifications

    """
    quality = []
    for field in tqdm(range(len(outputs)), desc = 'Inner Product IFTA Verification'):
        quality += [scalarProduct(target, normalize(outputs[field]))]
    if plotting:
        plt.plot(np.arange(1, len(quality)+1),quality, '.-')
        plt.title('Quality check from scalar product')
        plt.xlabel("Iterations")
        plt.ylabel("Normalized Scalar Product")
        plt.xscale('log')
        maxQuality = quality[-1]*100
        textstr = f'Final Quality = {maxQuality:.4g}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.5, 0.1, textstr,  fontsize=14,transform=plt.gca().transAxes,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()
    return quality


def InitializePropagator(inputBeam, z, wavelength = 253e-9, padding = 1,
                         extent = [-1.27 * 1e-2, 1.27 * 1e-2]):
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
    #Apply the padding to ensure high quality FFT
    paddedBeam = fastOn(inputBeam, padding)
    kBeam = fftshift(fft2(paddedBeam, workers = ncpu))
    kShape = kBeam.shape[0]
    

    # --- Step 2 Apply the propagator --- 
    #Creating k-Space coordinates
    kx_ = fftfreq(kShape, d = (2**(1+padding)) * extent[1]/kShape)
    ky_ = fftfreq(kShape, d = (2**(1+padding)) * extent[1]/kShape)
    
    kx_ = fftshift(kx_)
    ky_ = fftshift(ky_)
    kx, ky = np.meshgrid(kx_, ky_)
    kSquare = kx**2 + ky**2
    

    #Propagator taken from K Khare (see ref)
    propagator = np.exp(1j * z * np.sqrt(k0**2 - 4 * pi**2 * (kx**2 + ky**2)))
    
    return propagator

def InitializeLens(inputBeam, f, wavelength = 253e-9,
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

    #Built the lens transfer function using Serkez's radius of curvature implementation
    lensShift = np.exp(-1j * k0 * rSquare/(2 * f))
    
    return lensShift


def Gaussian(sizeFactor = 11, wavelength = 253e-9, w0 = 4e-3,
             extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    """
    Creates a Gaussian Beam for use in the Fresnel Optics Propagator

    Parameters
    ----------
    sizeFactor : int, optional
        The power that the size should be raised such that shape = 2^size. The default is 11.
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
    gridSize = 2 ** sizeFactor
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

