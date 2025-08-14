#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:44:28 2024

Setting up the limitations in our simulation of the phase plate 

@author: thomas
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from cmocean import cm
from numpy import pi
from PlottingTools import StanfordColormap
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
stanford_colormap = StanfordColormap()


def Box(inputPhase, group):
    """
    Performs grouped averaging on a phase mask to reduce its transverse resolution 
    (referred to as 'boxing').

    Parameters
    ----------
    inputPhase : np.array 
        2D array representing the phase mask that will be processed.
    group : int
        The size of the square pixel groups to average over. Both dimensions of the 
        phase mask must be divisible by this number.

    Raises
    ------
    IndexError
        The array should be a multiple of the group
        Generally should be a 2^n before or after shrinking

    Returns
    -------
    boxedPhase : np.array
        The phase mask with reduced resolution, where each group of pixels 
        has been averaged to represent a single phase value.

    """
    initialPhase = inputPhase
    
    # Adapting for a 2^n + 1 array shape
    if not ((initialPhase.shape[0]-1)%2):
        inputPhase = ShrinkOn(inputPhase)

    # Get the shape of the input array
    rows, cols = inputPhase.shape
    
    # --- Ensure the operation of grouping is possible ---
    if rows % group != 0 or cols % group != 0:
        raise IndexError("The input array is not able to be separated using the group Size")
    
    
    # Reshape the array to group the elements
    reshaped = inputPhase.reshape(rows // group, group, cols // group, group)

    # Sum within the groups
    group_sums = reshaped.sum(axis=(1, 3))
    
    # Expand the group sums back to the original shape
    expanded_sums = np.repeat(np.repeat(group_sums, group, axis=0), group, axis=1)
    
    # Divide by the group^2 to perform the average over the group
    boxedPhase = expanded_sums/group**2
    
    # Reshaping to 2^n+1 if necessary
    if not ((initialPhase.shape[0]-1)%2):
        boxedPhase = ShrinkOff(boxedPhase)

    return boxedPhase 



def ShrinkOn (inputPhase):
    """
    Removes the last column and row from the input phase to obtain a 2^n array
    
    Parameters
    ----------
    inputPhase : 2^n+1 array
        intial phase mask with a 2^n+1 size that needs to be reduced for grouping

    Returns
    -------
    2^n array 

    """
    return inputPhase[:-1,:-1]



def ShrinkOff (boxedPhase):
    """
    Extends the last row and column of the 2^n array to retrieve a 2^n+1 array
    The values that are extended outside of the lens are removed by the realLens function
    Parameters
    ----------
    boxedPhase : 2^n array 
        Grouped phase that needs to be extended back to 2^n+1 size


    Returns
    -------
    2^n+1 array

    """
    #Extend the last row of the array
    extended_row = np.vstack([boxedPhase, boxedPhase[-1, :]])

    #Extend the last column of the array
    extended_row = np.hstack([extended_row, extended_row[:, [-1]]])
    
    return extended_row


def Density(inputPhase, levels = 7, wavelength = 253e-9, materialIndex = 1.5058500198911624, n_air = 1.00030067):
    """
    This function sets the values of an array between -pi and pi to given equally spaced steps
    

    Parameters
    ----------
    inputPhase : array
        phase array defined from -pi to pi
    levels : int, optional
        determines the number of discrete steps the phase can take. The default is 7.
    wavelength : float, optional
        The laser's wavelength needed to determine material thickness. The default is 253e-9.
    materialIndex : float, optional
        Material's index of refraction needed to determine thickness. The default is 1.45 for UVFS at 253nm.

    Returns
    -------
    outputPhase : array
        array with defined discrete steps 
    depthStep : float
        difference in glass thickness for each phase step (m)

    """
    
    # Check that the operation is possible 
    assert levels > 0 
    
    # Determine the values of the phase that will be separated
    phaseDepth = np.linspace(-pi, pi , levels)
    outputPhase = np.array(inputPhase)
    # Set each phase point in the array to the closest level available
    phaseDifference = np.abs(phaseDepth[1] - phaseDepth[0])
    for phase in phaseDepth: 
        # Find where the array is closest to the given phase depth
        mask = (np.abs(np.subtract(inputPhase, np.ones(inputPhase.shape) * phase)) < phaseDifference/2+1e-9)
        # Apply the mask to set the phase at the given depth when in mask
        outputPhase[mask] = phase
    #print(np.unique(outputPhase)) # test that there are discrete levels
    
    
    # --- Calculating the depth of each phase step ---
    # Use the path length formula 
    depthStep = phaseDifference  * wavelength / (2*pi) / (materialIndex - n_air)
    
    return outputPhase, depthStep

def Density2Level(inputPhase, phaseValue, levels = 3, wavelength = 253e-9, materialIndex = 1.5058500198911624, n_air = 1.00030067):
    
    # Check that the operation is possible 
    assert levels > 0 
    
    # Determine the values of the phase that will be separated
    phaseDepth = np.linspace(-pi, pi , levels)
    outputPhase = np.array(inputPhase)
    # Set each phase point in the array to the closest level available
    phaseDifference = np.abs(phaseDepth[1] - phaseDepth[0])
    for phase in phaseDepth: 
        # Find where the array is closest to the given phase depth
        mask = (np.abs(np.subtract(inputPhase, np.ones(inputPhase.shape) * phase)) < phaseDifference/2+1e-9)

        # Apply the mask to set the phase at the given depth when in mask
        outputPhase[mask] = phase
    #print(np.unique(outputPhase)) # test that there are discrete levels
    
    # --- Replacing the central phase difference with the given phase value ---
    outputPhase[(outputPhase == phaseDepth[1])] = phaseValue
    #outputPhase[(outputPhase == phaseDepth[0])] = phaseDepth[2]
    #print(np.unique(outputPhase))
    phaseDifference = np.abs(phaseValue - phaseDepth[0])
    depthStep = phaseDifference  * wavelength / (2*pi) / (materialIndex - n_air)
    
    return outputPhase, depthStep



def maskToDepths(phase, wavelength = 253e-9, materialIndex = 1.5058500198911624, n_air = 1.00030067,
                 save = ''):
    
    
    # --- Identifying the full depth of the phase plate for a 2pi shift ---
    fullDepth = wavelength / (materialIndex-n_air)
    
    truePlate = np.array(phase)
    print(np.unique(truePlate))
    # --- Removing the unnecessary step -pi and setting to pi ---
    if np.array((truePlate == -pi)).sum() != 0:
        truePlate[(truePlate == -pi)] = truePlate[(truePlate == -pi)] + 2*pi
    else:
        print('The -pi phase was already removed')
    if np.array((truePlate < 0)).sum() != 0:
        truePlate[(truePlate < 0)] = truePlate[(truePlate < 0)] + 2*pi
    # --- Normalizing and setting to the depths instead of phase shift ---
    #truePlate += pi * np.ones(truePlate.shape)

    truePlate /=2*pi
    
    
    truePlate *= (-fullDepth)
    print(np.unique(truePlate))
    
    # --- Saving the True Phase Mask to a csv ---
    #np.savetxt(save, truePlate, delimiter=",", fmt="%.16f")
    #plt.imsave(save, truePlate, cmap="gray", format="png")
    plt.imshow(truePlate, cmap = 'gray')
    plt.show()
                     
    return truePlate
    

if __name__ == "__main__": 
    import RefractionIndex
    #--- Initial Test to see how the boxing works --- 
    
    # Extracting the classic Lenna Target for the test
    from Targets import Lenna
    testArray = np.ones([2**11, 2**11])
    testImage = Lenna(testArray, plot = False)
    testImage = np.array(testImage)

    # --- Applying Boxing ---
    result = Box(testImage, 16)
   
    # --- Plotting Outputs ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 10))
    
    # Importing the Stanford Colormap
    from PlottingTools import StanfordColormap
    stanford_colormap = StanfordColormap()
    
    # Plotting initial Image
    testImagePlot = axs[0].imshow(testImage, cmap = stanford_colormap)
    fig.colorbar(testImagePlot, ax = axs[0], orientation = 'horizontal')
    
    # Plotting grouped averaged image
    resultPlot = axs[1].imshow(result, cmap = stanford_colormap)
    fig.colorbar(resultPlot, ax = axs[1], orientation = 'horizontal')


    # --- Testing the Density Function --- 
    savePhase = 'Phase.h5'
    with h5py.File(savePhase, 'r') as file:
        hologram = file['Phase'][:]
    print('original phase steps', np.unique(hologram))
    #Testing on small array
    '''randomTest = np.random.rand(20,20) - 0.5
    randomTest *= 2*np.pi
    hologram = randomTest'''
    wavelength = 253e-9
    nLens = RefractionIndex.FusedSilica(wavelength)    #1.5058500198911624 # None
    nProp = RefractionIndex.Air(wavelength)  
    densePhase, depthStep = Density(hologram, levels = 8, wavelength=wavelength,
                                    n_air = nProp, materialIndex=nLens)
    #densePhase, depthStep = Density2Level(hologram, 0, levels = 3)
    print('Difference in glass thickness for each phase step', depthStep, '\n\n')

    figD, axsD = plt.subplots(1, 2, figsize=(12, 10))
    # Plotting initial Phase
    testImagePlotD = axsD[0].imshow(hologram, cmap = cm.curl)
    figD.colorbar(testImagePlotD, ax = axsD[0], orientation = 'horizontal')
    
    # Plotting grouped averaged image
    resultPlotD = axsD[1].imshow(densePhase, cmap = cm.curl)
    figD.colorbar(resultPlotD, ax = axsD[1], orientation = 'horizontal')
    
    
    # --- Making a manufacturable plate with true depth levels ---
    filepath = "Phase.png"
    truePlate = maskToDepths(hologram, save = filepath, wavelength=253e-9,
                             n_air = 1.0002950541290963, materialIndex=1.5014613746747054)
    
    figT, axsT = plt.subplots(1, 2, figsize=(12, 8))
    # Plotting initial Phase
    n_unique = len(np.unique(truePlate))
    #cmap = ListedColormap(get_cmap('tab20').colors[:n_unique])
    cmap = ListedColormap(cm.haline(np.linspace(0, 1, n_unique)))
    extent = np.array([-1.27e-2, 1.27e-2, -1.27e-2, 1.27e-2])*1e3
    testImagePlotT = axsT[0].imshow(truePlate, cmap = cmap, extent = extent)
    axsT[0].set_xlabel('x [mm]', fontsize = 14)
    axsT[0].set_ylabel('y [mm]', fontsize = 14)
    figT.colorbar(testImagePlotT, ax = axsT[0], orientation = 'horizontal')
    
    # Plotting grouped averaged image
    cropPlate = truePlate[int(truePlate.shape[0]/2 - 100) : int(truePlate.shape[0]/2 + 100),
                                           int(truePlate.shape[0]/2 - 100) : int(truePlate.shape[0]/2 + 100)]
    cropExtent = extent * cropPlate.shape[1]/truePlate.shape[1]
    resultPlotT = axsT[1].imshow(cropPlate, cmap = cmap, extent = cropExtent)
    
    axsT[1].set_xlabel('x [mm]', fontsize = 14)
    figT.colorbar(resultPlotT, ax = axsT[1], orientation = 'horizontal')
    figT.tight_layout()
    
    plt.show()
    boxed = Box(np.random.random(testArray.shape), 32)
    plt.imshow(boxed)
    plt.colorbar()
    plt.title('Boxing')
    plt.show()

    

    
        
    
    
