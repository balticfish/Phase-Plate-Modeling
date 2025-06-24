#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:48:11 2024

IFTA Algorithm using propagation through Open-MPI base

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
#from padding import fastOn, fastOff
def fastOn(inputArray, size):
    return inputArray
def fastOff(inputArray, size):
    return inputArray
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
import yaml

# MPI Implementation
from mpi4py import MPI
from mpi4py_fft import PFFT


# --- Globals ---
stanford_colormap = StanfordColormap()


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

    # --- Initialize the MPI Implementation --- 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # --- Verify Inputs ---
    N, M = inputField.shape
    assert N == M, "Field must be square"

    # ─── DOMAIN DECOMPOSITION: figure out which rows this rank owns ─────────
    counts = [N//n_ranks + (1 if r < N % n_ranks else 0) for r in range(n_ranks)]
    starts = np.cumsum([0] + counts[:-1])
    local_n = counts[rank]
    row0 = starts[rank]
    row1 = row0 + local_n
        

    # ─── INITIALIZE PHASE & AMPLITUDE ────────────────────────────────────────
    np.random.seed(randomSeed) #Setting a randomized seed for 
    phase_full = np.random.rand(N, M) * pi
    amp_full   = np.abs(inputField)

    # ─── SCATTER the real-space slabs to each rank ───────────────────────────
    if rank == 0:
        slabs = [phase_full[starts[r]:starts[r]+counts[r], :] for r in range(n_ranks)]
        amps  = [amp_full [starts[r]:starts[r]+counts[r], :] for r in range(n_ranks)]
    else:
        slabs = amps = None

    local_phase = comm.scatter(slabs, root=0)
    local_amp   = comm.scatter(amps,  root=0)
    
    
    # --- Initialising the target amplitude ---
    if target is None:
        target = superTruncGaussian(inputField, trunc = 50) #60 #returns an intensity
    target = np.sqrt(target)

    # ─── BROADCAST the target full array, then scatter it ────────────────────
    if rank == 0:
        tgt = np.sqrt(target)
        t_slabs = [tgt[starts[r]:starts[r]+counts[r], :] for r in range(n_ranks)]
    else:
        t_slabs = None

    local_target = comm.scatter(t_slabs, root=0)
    
    
    # --- Initialising the propagation matrices ---
    # This saves on computation time for each iteration
    if rank == 0:
        # Initialising the propagator k space for after lens
        propagatorForward = ifftshift(InitializePropagator(inputField, z, padding = size, nProp = nProp,
                                                 extent = extent, wavelength = wavelength))
        propagatorBackward = ifftshift(InitializePropagator(inputField, -z, padding = size, nProp = nProp,
                                                  extent = extent, wavelength = wavelength))
        # Initialising the lens meshgrid
        lensShiftForward = InitializeLens(inputField, f, nLens = nLens, nProp = nProp,
                                          extent = extent, wavelength = wavelength)
        lensShiftBackward = InitializeLens(inputField, -f, nLens = nLens, nProp = nProp,
                                           extent = extent, wavelength = wavelength)
        # Initialising the propagator k space for before lens (in lens)
        if z0 != None:
            propagatorForward0 = ifftshift(InitializePropagator(inputField, z0, padding = size, nProp = nLens,
                                                      extent = extent, wavelength = wavelength))
            propagatorBackward0 = ifftshift(InitializePropagator(inputField, -z0, padding = size, nProp = nLens,
                                                       extent = extent, wavelength = wavelength))
    else:
        # Avoid Broadcasting when not in root 0
        propagatorForward, propagatorBackward, lensShiftForward, lensShiftBackward = [None]*4
         
        # Initialising the propagator k space for before lens (in lens)
        if z0 != None:
            propagatorForward0, propagatorBackward0 = [None]*2

    # --- Broadcasting the propagators for MPI ---
    propagatorForward   = comm.bcast(propagatorForward, root=0)
    propagatorBackward  = comm.bcast(propagatorBackward, root=0)
    propagatorForward0  = comm.bcast(propagatorForward0, root=0)
    propagatorBackward0 = comm.bcast(propagatorBackward0, root=0)
    lensShiftForward    = comm.bcast(lensShiftForward, root=0)
    lensShiftBackward   = comm.bcast(lensShiftBackward, root=0)

    # --- Slice the propagators down to the local slab ---
    pad_factor = propagatorForward.shape[0] // N
    pad_factor = 1
    row0_pad   = row0 * pad_factor
    row1_pad   = row1 * pad_factor
    
    local_propF = propagatorForward[:, row0_pad:row1_pad]
    local_propB = propagatorBackward[:, row0_pad:row1_pad]
    local_lensF = lensShiftForward[row0:row1, :]
    local_lensB = lensShiftBackward[row0:row1, :]
    if z0 != None:
        local_propF0 = propagatorForward0[:, row0_pad:row1_pad]
        local_propB0 = propagatorBackward0[:, row0_pad:row1_pad]
    


    # ─── SET UP THE DISTRIBUTED FFT PLAN ─────────────────────────────────────
    fft_MPI = PFFT(comm, propagatorForward.shape, axes=(0,1), dtype=np.complex128)
    
    
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

    if save and rank ==0:
        saveFile = h5py.File(save, 'w')
        outputFP = saveFile.create_dataset(
            'OutputFP',
            shape = (iteration, inputField.shape[0], inputField.shape[1]),
            chunks = (1, inputField.shape[0], inputField.shape[1])
            #dtype = 'float64'
            )
    else:
        saveFile = None
        outputFP = None


    # ─── MAIN IFTA LOOP ──────────────────────────────────────────────────────
    for i in tqdm(range(iteration), desc = 'Iterations of IFTA'):
        
        # --- Input the initial Intensity and keep the varying phase --- 
        local_field = local_amp * np.exp(1j * local_phase)
        
        # --- Single Lens Optic Propagation ---
        if z0 != None:
            # Propagate the beam to the lens
            pad_slab = fastOn(local_field, size)
            k_slab   = fft_MPI.forward(pad_slab)
            k_slab  *= local_propF0
            out_slab = fft_MPI.backward(k_slab)
            local_field = fastOff(out_slab, size)
        
        # --- Propagate to Fourier Plane --- 
        # Apply the forward lens shift to the beam
        local_field *= local_lensF
        
        # Propagate the beam to the lens' Focal Plane 
        pad_slab    = fastOn(local_field, size)
        k_slab      = fft_MPI.forward(pad_slab)
        k_slab     *= local_propF
        out_slab    = fft_MPI.backward(k_slab)
        local_field = fastOff(out_slab, size)
        
        # --- Extracting Analytics --- 
        if save:
            slabFP = np.abs(local_field)**2
            gathered = comm.gather(slabFP, root=0)
            if rank == 0:
                outputFP[i,:,:] = np.vstack(gathered)
                saveFile.flush()
    
        
        
        # --- Replace the intensity with the target intensity --- 
        local_phase = np.angle(local_field)
        local_field = local_target * np.exp(1j * local_phase)
        
        # --- Transform back to Carthesian --- 
        
        # Undoing the propagation
        pad_slab     = fastOn(local_field, size)
        k_slab       = fft_MPI.forward(pad_slab)
        k_slab      *= local_propB
        out_slab     = fft_MPI.backward(k_slab)
        local_field  = fastOff(out_slab, size)
        local_field *= local_lensB
        
        # --- Single Lens Optic Propagation ---
        if z0 != None:
            # Propagate the beam to the phase plate
            pad_slab    = fastOn(local_field, size)
            k_slab      = fft_MPI.forward(pad_slab)
            k_slab     *= local_propB0
            out_slab    = fft_MPI.backward(k_slab)
            local_field = fastOff(out_slab, size)
        
        # --- Collect the new phase --- 
        local_phase = np.angle(local_field)
                
        # --- Adjusting the phase to error --- 
        '''if rank == 0:
            full_phase = np.vstack(parts)
            if steps != None: #Sets the phase to discrete levels through rounding error
                phase, thickness = Density(phase, levels = steps, wavelength=wavelength, n_air=nProp, materialIndex=nLens)
                if steps == 3:
                    phase, thickness = Density2Level(phase, 0, levels = steps, wavelength=wavelength, n_air=nProp, materialIndex=nLens)
                
            if boxing !=None: #Averages over given pixel size for transverse constraints
                phase = Box(phase, boxing)
            if realLens != None: #Sets the phase to 0 outside the lens dimensions
                phase[outsideLens] = 0
        '''
        gc.collect()


        
    all_phases = comm.gather(local_phase, root=0)


    
    # --- If plot is set to True the outputs will be neatly plotted ---
    '''if plot:
        # --- Plotting the Fourier Plane Intensity --- 
        
        all_iter = comm.gather(local_phase, root=0)
        if rank == 0:
            outputIter = np.vstack(all_iter)
            plt.imshow(outputIter, cmap = stanford_colormap, extent = [extent[0], extent[1], extent[0], extent[1]])#, norm = LogNorm())
            plt.title('Fourier Plane Output Intensity')
            plt.xlabel('distance (m)')
            plt.ylabel('distance (m)')
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            
            # --- Plotting the hologram -> Phase Distribution --- 
            outputPhase = np.vstack(all_phases)
            plt.imshow(outputPhase, cmap = cm.curl, extent = [extent[0], extent[1], extent[0], extent[1]])
            plt.title('Post-Phase-Plate Beam Phase')
            plt.xlabel('Distance (m)', fontsize = 14)
            plt.ylabel('Distance (m)', fontsize = 14)
            plt.colorbar()
            plt.tight_layout()
            plt.show()
    ''' 
    
    
    if rank == 0:
        phase = np.vstack(all_phases)
    else:
        phase = None
    if rank == 0 and save:
        saveFile.create_dataset('Phase', data=phase)
        saveFile.create_dataset('Target', data=target)
        saveFile.close()
        
        print(f' --- Phase Mask Saved under {save} --- ')
    
        
    # --- If set the SSE will be computed and plotted if plot is asked --- 
    if SSE:
        # --- Plotting the quality of the output throughout the process --- 
        if save and rank == 0:
            with h5py.File(save, 'r') as file:
                fpStack = file['OutputFP'][...]    
            SSE_result = SSEPlotting(fpStack, target, plotting = False)
            del fpStack
            gc.collect()
            with h5py.File(save, 'a') as file:
                file.create_dataset('SSE', data=SSE_result)
                
    return phase



if __name__ == "__main__":
    
    # ––– Importing the Globals through the yaml file ––––––––––––––––––––
    with open("IFTA.yaml") as f:
        cfg = yaml.safe_load(f)
    extent = np.array(cfg['extent'])
    wavelength = float(cfg['wavelength'])
    inputWidth = float(cfg.pop("inputWidth"))
    # ––– Creating an input Gaussian to test the IFTA ––––––––––––––––––––
    # --- Testing the IFTA ---
    z0 = pi/wavelength * inputWidth**2
    q0 = 1j * z0
    k0 = 2* pi / wavelength
    
    # --- Build a meshgrid to apply the gaussian too ---
    gridSize = 2**12 +1
    x_ = np.linspace(extent[0], extent[1], gridSize)
    y_ = np.linspace(extent[0], extent[1], gridSize)
    X, Y = np.meshgrid(x_, y_)
    rSquare = X**2 + Y**2
    
    # --- Creating the gaussian field using complex beam parameter ---    
    field = 1/q0 * np.exp(- 1j * k0 * rSquare / (2 * q0))
    #plt.imshow(np.abs(field)**2)
    #plt.show()
    
    # ––– Initializing a target ––––––––––––––––––––––––––––––––––––––––––
    target = flatTop(field, extent = extent, w0 = 6e-4, plot = False)
    
    # --- Finding the phase necessary for transformation ---
    '''phase = IFTA(inputBeam, iteration = 30, target = target,
                       plot = True, SSE = True, steps = None,
                       save = 'Test_IFTA_MPI.h5', realLens = None, boxing = None, z0 = z1,
                       f = f, z = f, randomSeed = randomSeed, nProp = nProp, nLens = nLens,
                       extent = extent, size = 0, wavelength = wavelength)
    '''
    
    phase = IFTA(field, 
                 target = target, 
                 **{k: v for k, v in cfg.items()}
                )
    
