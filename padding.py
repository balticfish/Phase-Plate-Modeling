#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:18:15 2024

@author: thomas
"""

import numpy as np

def paddingOn(inputArray, size):
    """
    Returns a padded array to allow for more precise FFTs

    Parameters
    ----------
    inputArray : np.array
        input image that should be padded
    size : int
        The padded array will be (size + 1)x bigger in shape

    Raises
    ------
    ValueError
        When the input shape is odd as this causes wrong centering of the array

    Returns
    -------
    paddedOut : np.array
        input array centered with zeroes elsewhere
    """
    
    # --- Determine the shape of the input ---
    inputShape = inputArray.shape
    
    # --- Raise an error for odd shapes to make sure the input is centered ---
    #Could go around this by using NaNs and identification later but much more work
    if inputShape[0] % 2 != 0 or inputShape[1] % 2 != 0:
        raise ValueError ('The input shape has an odd number disallowing centering')
        return
        
        
    # --- Prepare the new padded array --- 
    paddedShape = (inputShape[0] * (1 + size), inputShape[1] * (1 + size))
    paddedOut = np.zeros(paddedShape, dtype=inputArray.dtype)
    
    startRow = int((paddedShape[0] - inputShape[0]) / 2)
    startCol = int((paddedShape[1] - inputShape[1]) / 2)

    # --- Center the Input Array in the padded one ---
    paddedOut[startRow:startRow + inputShape[0], startCol:startCol + inputShape[1]] = inputArray
 
    return paddedOut

def paddingOff(paddedArray, size):
    """
    Removing the padding created through the paddingOn function above
    This allow for fast computing time after FFT modifications are done

    Parameters
    ----------
    paddedArray : np.array
        array whose padding should be removed 
    size : int 
        Should be the same int as when created with paddingOn

    Returns
    -------
    unpadded : np.array
        Image before the padding was added by paddingOn
    """
    
    # --- Determine the shape of the padded array ---
    paddedShape = paddedArray.shape
    
    # --- Extract the unpadded array --- 
    extractShape = np.array(paddedShape)/(1+size)
    
    startRow = int((paddedShape[0] - extractShape[0]) / 2)
    startCol = int((paddedShape[1] - extractShape[1]) / 2)
    unpadded = paddedArray[startRow:(startRow + int(extractShape[0])), startCol:(startCol + int(extractShape[1]))]
    
    return unpadded

def fastOn(inputArray, size):
    """
    Applies a padding to the input array such that it maximises computing time for an FFT
    Increases the input Array with a padding such that 2^n -> 2^(n+size)

    Parameters
    ----------
    inputArray : np.array
        The array that should be padded with shape 2^n x 2^n
    size : integer
        size factor increase

    Raises
    ------
    ValueError
        The input array should be a factor of 2 (initial fast shape)
        The input array should be symmetrical

    Returns
    -------
    paddedOut : np.array
        padded array such that the input is centered in a 2^(n+size) array
        surrounded by 0s

    """
    # --- Determine the shape of the current Array ---
    inputShape = inputArray.shape
    factor = np.log2(inputShape[0] - 1)
    
    # --- Catching Errors for Fast Transforms ---
    if factor.is_integer() == False:
        raise ValueError("The input Array was not in a fast shape 2**n + 1")
    if inputShape[0] != inputShape[1]:
        raise ValueError("The input Array should be symmetrical: 2**n+1 x 2**n+1")
    
    # --- Prepare the new padded shape ---
    newFactor = factor + size
    
    padding = int((2**newFactor - 2**factor)/2)
    paddedOut = np.pad(inputArray, padding)
    
    return paddedOut

def fastOff(paddedArray, size): 
    """
    To be applied after a fastOn, 
    Removes the padding applied by fastOn given the same size factor

    Parameters
    ----------
    paddedArray : np.array
        Input array with padding added from fastOn
    size : integer
        factor that the array was padded using fastOn

    Returns
    -------
    unpadded : np.array
        Array with removed padding from fastOn

    """
    
    # --- Determine the shape of the padded array ---
    paddedShape = paddedArray.shape
    #print(paddedShape)
    paddedFactor = np.log2(paddedShape[0] - 1)
    #print(paddedFactor)
    # --- Determine the size of the unpadded array ---
    extractShape = 2**(paddedFactor - size)+1
    
    # --- Extract the unpadded array ---
    startRow = ((paddedShape[0] - extractShape) / 2).astype(int)
    startCol = ((paddedShape[1] - extractShape) / 2).astype(int)

    unpadded = paddedArray[startRow :(startRow + int(extractShape)), startCol:(startCol + int(extractShape))]
    return unpadded

def fastOn_2n(inputArray, size):
    """
    Applies a padding to the input array such that it maximises computing time for an FFT
    Increases the input Array with a padding such that 2^n -> 2^(n+size)

    Parameters
    ----------
    inputArray : np.array
        The array that should be padded with shape 2^n x 2^n
    size : integer
        size factor increase

    Raises
    ------
    ValueError
        The input array should be a factor of 2 (initial fast shape)
        The input array should be symmetrical

    Returns
    -------
    paddedOut : np.array
        padded array such that the input is centered in a 2^(n+size) array
        surrounded by 0s

    """
    # --- Determine the shape of the current Array ---
    inputShape = inputArray.shape
    factor = np.log2(inputShape[0])
    # --- Catching Errors for Fast Transforms ---
    if factor.is_integer() == False:
        raise ValueError("The input Array was not in a fast shape 2**n")
    if inputShape[0] != inputShape[1]:
        raise ValueError("The input Array should be symmetrical: 2**n x 2**n")
    
    # --- Prepare the new padded shape ---
    newFactor = factor + size
    paddedShape = 2**newFactor
    paddedOut = np.zeros((int(paddedShape),int(paddedShape)), dtype = inputArray.dtype)
    
    # --- Centering the input Array inside the padded one --- 
    
    startRow = int((paddedShape - inputShape[0]) / 2)
    startCol = int((paddedShape - inputShape[1]) / 2)
    paddedOut[startRow:startRow + inputShape[0], startCol:startCol + inputShape[1]] = inputArray
    
    return paddedOut

def fastOff_2n(paddedArray, size): 
    """
    To be applied after a fastOn, 
    Removes the padding applied by fastOn given the same size factor

    Parameters
    ----------
    paddedArray : np.array
        Input array with padding added from fastOn
    size : integer
        factor that the array was padded using fastOn

    Returns
    -------
    unpadded : np.array
        Array with removed padding from fastOn

    """
    
    # --- Determine the shape of the padded array ---
    paddedShape = paddedArray.shape
    paddedFactor = np.log2(paddedShape[0])
    
    # --- Determine the size of the unpadded array ---
    extractShape = 2**(paddedFactor - size)
    
    # --- Extract the unpadded array ---
    startRow = int((paddedShape[0] - extractShape) / 2)
    startCol = int((paddedShape[1] - extractShape) / 2)
    unpadded = paddedArray[startRow:(startRow + int(extractShape)), startCol:(startCol + int(extractShape))]
    return unpadded
    
    
    
if __name__ == "__main__":
    # Create a 9x9 array with increasing integers
    array_9x9 = np.arange(1, 26).reshape(5,5)
    print(array_9x9)
    fastInt = 11
    faster = fastOn(array_9x9, fastInt)
    print(faster, faster.shape,'\n\n')
    slower = fastOff(faster, fastInt)
    
    print(slower, '\n\n')










































