#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:12:02 2025

Stanford ColorMap and other Plotting tools

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from cmocean import cm as cmo
from matplotlib.ticker import ScalarFormatter
import h5py
from scipy.optimize import curve_fit
from scipy.special import erf



def StanfordColormap():
    
    # Define the Stanford colors
    stanford_colors = {
        'dark_blue': (0.0, 124/255, 146/255),  # Lagunita
        'cyan': (66/255, 152/255, 181/255),    # Sky
        'bay_green': (111/255, 162/255, 135/255),  # Bay
        'green': (39/255, 153/255, 137/255),   # Palo Verde
        'yellow': (254/255, 221/255, 92/255),  # Illuminating
        'red': (0.69, 0.1, 0.12)               # Stanford Red
    }
    
    # Define the key color positions in the colormap
    positions = [0.0, 0.2, 0.4, 0.8, 1.0]
    
    # Define the colors corresponding to each position
    color_list = [
        stanford_colors['dark_blue'],
        stanford_colors['cyan'],
        stanford_colors['green'],
        stanford_colors['yellow'],
        stanford_colors['red']
    ]
    
    # Create a LinearSegmentedColormap
    stanford_colormap = LinearSegmentedColormap.from_list("stanford_jet", list(zip(positions, color_list)))
    return stanford_colormap

def Centroid(im):
    # Find the indices of all non-zero elements
    
    im = im/np.sum(im)
    
    # Get the corresponding intensities
    non_zero_indices = np.argwhere(im > im/100)
    intensities = im[im > im/100]
    
    
    # Calculate the total intensity
    total_intensity = np.sum(intensities)

    # Calculate the weighted sum of the indices
    weighted_sum = (non_zero_indices * intensities[:, np.newaxis]).sum(axis=0)
    
    # Calculate the centroid coordinates
    centroid_row, centroid_col = weighted_sum / total_intensity
    return int(round(centroid_row)), int(round(centroid_col))




def generateData(r, sigma, a):
    np.random.seed(0)
    yTrue = truncGaussianModel(r, sigma, a)
    yData = yTrue + 0.01 * np.random.normal(size=len(r))
    return yData, yTrue





def plotBeam(field, extent = [1.27e-2, 1.27e-2], fitCut = True, title = "Fresnel Beam after Propagation",
             truncRadius = 6e-4, maxROI = 220):
    
    # --- Extract the Globals --- 
    stanford_colormap = StanfordColormap()
    pixelSize = 2 * extent[1] / field.shape[-2]
    
    # --- Centering the beam and reducing the ROI ---
    #Should keep the grid unchanged aka not moving the 0 of the grid 
    
    # Extract the centroid position
    xCentroid, yCentroid = Centroid(np.abs(field)**2) 

    '''# Extract the vertical and horizontal cuts
    fieldHorizontal, fieldVertical = field[int(xCentroid), :], field[:, int(yCentroid)] 
    
    # Determine the ROI 
    normIntensity = np.abs(field)**2 / np.sum(np.abs(field)**2)
    minValueV, minValueH = np.max(np.abs(fieldVertical)**2), np.max(np.abs(fieldHorizontal)**2)

    maskPlot = (np.abs(fieldVertical)**2 / np.sum(np.abs(fieldVertical)**2) > minValueV) | (
        np.abs(fieldHorizontal)**2 / np.sum(np.abs(fieldHorizontal)**2) > minValueH)

    maskTruth = np.sum(maskPlot)'''

    #maxROI = np.min([xCentroid, yCentroid, maskTruth])
    #maxROI = 220
    fieldROI = field[int(xCentroid-maxROI): int(xCentroid+maxROI), int(yCentroid-maxROI):int(yCentroid+maxROI)]
    
    


    
    # --- Extract the transverse Intensity cut --- 
    cutIntensityROI = np.abs(field[int(xCentroid-maxROI): int(xCentroid+maxROI), int(yCentroid)])**2
    norm = np.sum(cutIntensityROI)
    # --- Determine the FWHM | 1/e2 of the beam --- 
    
    
    # --- Initialising the fit functions ---
    def fitData(xSpace, yData):
        """
        Fit the erf_model to the provided data.

        Parameters:
        x_data : array-like
            Input data.
        y_data : array-like
            Output data to fit.

        Returns:
        popt : array
            Optimized parameters.
        pcov : 2D array
            Covariance of the optimized parameters.
        """
        initial_guess = [0.509593e-3]  # Initial parameter guesses
        popt, pcov = curve_fit(truncGaussianModel, xSpace, yData, p0=initial_guess)
        return popt, pcov
    
    def truncGaussianModelSlice(r, sigma):
        '''
        Truncated Gaussian Intensity formula

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.
        a : TYPE, optional
            DESCRIPTION. The default is 0.6e-3.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        a = truncRadius
        if truncRadius == None:
            a = 0.6e-3

        A_T = 1/(sigma * (erf(a/(np.sqrt(2)*sigma)) - erf(-a/(np.sqrt(2)*sigma)))) * np.sqrt(2/np.pi)
        model = A_T * np.exp(-r**2 / (2*sigma**2)) 

        return model

    def truncGaussianModel(r, sigma):
        '''
        Truncated Gaussian Intensity formula

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.
        a : TYPE, optional
            DESCRIPTION. The default is 0.6e-3.

        References:
        Alex Halavanau's derivation of the the analytical Truncated Gaussian 
        
        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        a = truncRadius
        if truncRadius == None:
            a = 0.6e-3
        
        A_T = 1/(sigma * (erf(a/(np.sqrt(2)*sigma)) - erf(-a/(np.sqrt(2)*sigma)))) * np.sqrt(2/np.pi)
        model = A_T * np.exp(-r**2 / (2*sigma**2)) 

        return model / np.sum(model)
    
    
    # --- Fitting the data to the trunc-Gaussian Model ---
    if fitCut == True:
        # Creating x values
        truncROI = int(truncRadius / pixelSize)
        xSpaceTrunc = np.linspace(-truncRadius, truncRadius, int(2*truncROI+1))

        
        cutIntensityTruncROIFit = np.abs(field[int(xCentroid-truncROI):int(xCentroid+truncROI+1), int(yCentroid)])**2
        #Changing the normalization to match the fit
        norm = np.sum(cutIntensityTruncROIFit)
        cutIntensityROINormFit = cutIntensityTruncROIFit / norm

        # Creating a truncGaussian Fit
        popt, pcov = fitData(xSpaceTrunc, cutIntensityROINormFit)
        print(popt, pcov)
        model = truncGaussianModelSlice(xSpaceTrunc, *popt)
    
        # --- Determine the cut ---
        sigma = popt[0]
        print(sigma)
        a = truncRadius
        A_T = 1/(sigma * (erf(a/(np.sqrt(2)*sigma)) - erf(-a/(np.sqrt(2)*sigma)))) * np.sqrt(2/np.pi)

        cutPercent = model[-1] / A_T * 100
        uncertaintyCP = np.abs(cutPercent * truncRadius**2 / sigma **3 * np.sqrt(pcov[0][0]))
        print('Cut Percent', cutPercent, '±', uncertaintyCP)

        #cutPercent = np.min(model) / np.max(model) * 100
        
        
        
    
    # --- Plotting ---
    # Extracting the necessary parameters

    plotExtent = [-pixelSize * (maxROI + (field.shape[0]/2 - xCentroid)), pixelSize * (maxROI + (field.shape[0]/2 - xCentroid))]

    # Creating the figure 
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.5])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,:])
    
    # Plotting the Phase of the beam
    Phase = ax1.imshow(np.angle(fieldROI), cmap = cmo.curl,
                      extent = [plotExtent[0], plotExtent[1], plotExtent[0], plotExtent[1]])
    
    ax1.set_title("Phase")
    fig.colorbar(Phase, ax = ax1, orientation = 'horizontal')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.set_xlabel('Beam size (m)')
    ax1.set_ylabel('Beam size (m)')
    
    # Plotting the Amplitude of the beam
    Amplitude = ax2.imshow(np.abs(fieldROI), cmap = cmo.curl
                      , extent = [plotExtent[0], plotExtent[1], plotExtent[0], plotExtent[1]])
    ax2.set_title("Amplitude")
    fig.colorbar(Amplitude, ax = ax2, orientation = 'horizontal')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.set_xlabel('Beam size (m)')
    
    # Plotting the Intensity of the beam
    Intensity = ax3.imshow(np.abs(fieldROI)**2, cmap = stanford_colormap,
                           extent = [plotExtent[0], plotExtent[1], plotExtent[0], plotExtent[1]])
    ax3.set_title("Intensity")
    fig.colorbar(Intensity, ax = ax3, orientation = 'horizontal')
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.set_xlabel('Beam size (m)')
    
    # Plotting the Cut Intensity Profile of the beam
    xPlot = np.linspace(plotExtent[0], plotExtent[1], len(cutIntensityROI))
    cutIntensityROINorm = cutIntensityROI/norm
    ax4.plot(xPlot, cutIntensityROINorm)
    #ax4.plot(xSpaceTrunc, cutIntensityROINormFit) #Changed briefly for testing

    # Formatting the axes
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-4, 4))
    ax4.xaxis.set_major_formatter(formatter)
    ax4.yaxis.set_major_formatter(formatter)
    ax4.set_title("Center cut of the Intensity of the propagated beam")
    ax4.set_xlabel('Beam size (m)')
    ax4.set_ylabel('Normalized Intensity')
    
    if fitCut == True:
        # Overlaying the cut-Gaussian fit 
        ax4.plot(xSpaceTrunc, model/np.sum(model), 'r--')
        
        # Adding a text box giving the cutPercentage for the truncGaussian Fit ---
        cutString = f'Cut Percentage = {cutPercent:.3g}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.75, 0.9, cutString, transform=ax4.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        
        
    # --- Extra decoration ---
    # Plotting the expected truncation radius
    if truncRadius != None:
        ax4.axvline(-truncRadius,linestyle = '--', color = 'k', alpha = 0.5)
        ax4.axvline(truncRadius,linestyle = '--', color = 'k', alpha = 0.5)
    
    # Adding a title
    fig.suptitle(title, size = 20)
    
    # Extra Arrangements
    fig.tight_layout()
    return


def superTruncGaussian(inputBeam, targetRadius =(1.2e-3)/2, n = 1, trunc = 50,
                       extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    """
    Generates an array with the intensity pattern of a truncated super Gaussian beam transverse profile
    This function is used in association with GSA to be used as a target
    
    Parameters
    ----------
    inputBeam : np.array
        This input is used to set the size of the output target
    w0 : float, optional
        DESCRIPTION. The default is set in the globals.
    n : float, optional
        The power the Gaussian should be raised to before truncation. 
        The default is 1 no increase in power
    trunc : float, optional
        The truncation as a percentage of the beam waist w0. The default is None for no truncation
        For instance trunc = 80 would set the values R > 0.8 * w0 to 0
    extent : np.array, optional
        Extent of the array to build. The default is set in the globals.
    plot : Bool, optional
        Decides if outputs should be plotted. The default is False.

    Returns
    -------
    intensity : np.array
        Transverse Intensity pattern of a Truncated Super Gaussian Beam

    """
    
    shape = inputBeam.shape[0]
    
    x_ = np.linspace(extent[0], extent[1], shape)
    y_ = np.linspace(extent[0], extent[1], shape)
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    sigma = targetRadius / (2*np.sqrt(2*np.log(100/trunc))) * 2 * np.sqrt(2)
    intensity = np.abs(np.exp(-(rSquare/ (2*sigma**2))**n)) **2
    if trunc != None: 
        intensity[intensity < trunc * np.max(intensity) / 100] = 0
        area = np.sum((intensity>0)) # (intensity>1e-5)
        radius = np.sqrt(area / np.pi) * (2*extent[1]/intensity.shape[1])
        print('Target Diameter:', 2 * radius * 1e3 , 'mm')
    
    if plot:
        plt.imshow(intensity, extent = [extent[0], extent[1], extent[0], extent[1]], cmap = cmo.curl)
        #plt.title("Cut-Guassian Target")
        plt.xlabel('Distance (m)', size = 13)
        plt.ylabel('Distance (m)', size = 13)
        plt.tight_layout()
        plt.colorbar()
        plt.show()
        
    return intensity



if __name__ == "__main__":
    
    
    # --- Testing the Truncated Gaussian Fit --- 
    testArray = np.ones([2**12+1, 2**12+1])
    extent = [-1.27e-2, 1.27e-2]
    
    targetRadius = (1.2/2)*1e-3 # Radius of the target that should be achieved after propagation
    trunc = 50
    test = superTruncGaussian(testArray, targetRadius = targetRadius, n = 1, trunc = trunc, 
                           extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = True)
    test += (np.random.rand(testArray.shape[0], testArray.shape[1]))*1e-1
    plotBeam(np.sqrt(test), title='Truncation test', fitCut = True, truncRadius= targetRadius, extent = extent)
    
    xSpace = np.linspace(-0.6e-3, 0.6e-3, num = 97)
    modelFit = truncGaussianModel(xSpace, 0.509593e-3)


