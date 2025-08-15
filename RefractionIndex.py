#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 08:01:43 2025

Functions to determine the index of refraction at specific wavelengths

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

def FusedSilica(wavelength, T_C = 20):
    """
    Temperature-aware refractive index of fused silica (SiO2)
    using Malitson Sellmeier + thermo-optic correction.

    Parameters
    ----------
    wavelength : float
        Wavelength in meters
    T_C : float
        Temperature in Celsius (default: 20°C)

    Returns
    -------
    n : float
        Refractive index at given T and λ
    """
    # Convert wavelength from m to micrometers
    wavelength_um = wavelength * 1e6
    lambda2 = wavelength_um**2
    
    # --- Malitson Sellmeier (1965) --- 
    term1 = 0.6961663 * lambda2 / (lambda2 - 0.0684043**2)
    term2 = 0.4079426 * lambda2 / (lambda2 - 0.1162414**2)
    term3 = 0.8974794 * lambda2 / (lambda2 - 9.896161**2)
    n2_minus_1 = term1 + term2 + term3
    n = np.sqrt(n2_minus_1 + 1)
    
    # Thermo-optic correction
    dn_dT = 1e-5  # [1/K]
    delta_T = T_C - 20.0
    n_T = n + dn_dT * delta_T
    
    return n_T

def CalciumFluoride(wavelength, T_C = 20):
    """
    Temperature-aware refractive index of CaF2 (Calcium Fluoride)
    using Zelmon Sellmeier + linear dn/dT correction.

    Parameters
    ----------
    wavelength : float
        Wavelength in meters
    T_C : float
        Temperature in Celsius (default: 20°C)

    Returns
    -------
    n : float
        Refractive index at given T and λ
    """
    wavelength_um = wavelength * 1e6
    lambda2 = wavelength_um**2
    
    # --- Zelmon Sellmeier (1998) --- 
    n2_minus_1 = (0.5675888 * lambda2 / (lambda2 - 0.050263605**2) +
              0.4710914 * lambda2 / (lambda2 - 0.1003909**2) +
              3.8484723 * lambda2 / (lambda2 - 34.649040**2))
    n = np.sqrt(n2_minus_1 + 1)
    
    # --- Thermo-optic correction --- 
    dn_dT = 1.5e-6  # [1/K]
    delta_T = T_C - 20.0
    n_T = n + dn_dT * delta_T
    
    return n_T


def Air(wavelength, T_C=20.0, p=101325, RH=0.5):
    """
    Computes the refractive index of moist air at arbitrary T, p, and RH.
    Uses Birch & Downs (1994) for UV and Ciddor's method for water vapor.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters (200–1700 nm range)
    T_C : float
        Temperature in °C
    p : float
        Total pressure in Pa
    RH : float
        Relative humidity [0–1]

    Returns
    -------
    n_air : float
        Refractive index of moist air
    """
    λ_um = wavelength * 1e6
    sigmaSquare = (1 / λ_um)**2
    T_K = T_C + 273.15
    
    # --- Birch & Downs for dry air ---
    n_dry = 1 + 1e-8 * (
        8342.54 + 2406147 / (130 - sigmaSquare) + 15998 / (38.9 - sigmaSquare)
    )
    
    # --- Water vapor contribution (Ciddor-style) ---
    n_water = 1 + 1.022e-8 * (
        (295.235 + 2.6422 * sigmaSquare + 0.03238 * sigmaSquare**2 + 0.004028 * sigmaSquare**3)
        / (1 - 0.01041 * sigmaSquare)
    )
    
    # --- Saturation vapor pressure (Buck 1981) ---
    e_s = 6.1121 * np.exp((17.502 * T_C) / (240.97 + T_C))  # hPa
    e_s *= 100  # Pa
    p_w = RH * e_s
    p_d = p - p_w
    
    # --- Combined index ---
    n_air = 1 + (p_d / 101325) * (n_dry - 1) + (p_w / 101325) * (n_water - 1)
    return n_air

def plot_refractive_index():
    # Define wavelength ranges
    wavelengths_full = np.linspace(200, 7000, 1000)*1e-9
    wavelengths_zoomed = np.linspace(200, 800, 1000) *1e-9

    # Calculate refractive indices
    refractive_indices_full = FusedSilica(wavelengths_full)
    refractive_indices_zoomed = FusedSilica(wavelengths_zoomed)
    
    refractive_indices_full_CaF2 = CalciumFluoride(wavelengths_full)
    refractive_indices_zoomed_CaF2 = CalciumFluoride(wavelengths_zoomed)
    refractive_indices_zoomed_Air = Air(wavelengths_zoomed)
    # Calculate the index of refraction for 253 nm
    wavelength_253 = 253e-9
    n_253 = FusedSilica(wavelength_253)
    print(n_253)
    n_253_CaF2 = CalciumFluoride(wavelength_253)
    wavelength_ThorLabs = 588e-9
    nThorLabs = FusedSilica(wavelength_ThorLabs)
    nThorLabs_CaF2 = CalciumFluoride(wavelength_ThorLabs)
    
    # Create the main plot
    fig, ax_main = plt.subplots(figsize=(12, 8))

    # Plot the zoomed range (200 - 800 nm) in the main plot
    ax_main.plot(wavelengths_zoomed, refractive_indices_zoomed, label='Fused Silica', color='red')
    ax_main.plot(wavelengths_zoomed, refractive_indices_zoomed_CaF2, label='Calcium Fluoride', color='Blue')

    ax_main.text(440e-9, 1.53, f'Fused Silica \nn at 253nm = {n_253:.6f}\nn at 588nm = {nThorLabs:.6f}', fontsize=12, ha='center', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax_main.text(440e-9, 1.51, f'Calcium Fluoride \nn at 253nm = {n_253_CaF2:.6f}\nn at 588nm = {nThorLabs_CaF2:.6f}', fontsize=12, ha='center', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Labels and title for the main plot
    ax_main.set_xlabel('Wavelength (nm)')
    ax_main.set_ylabel('Refractive Index (n)')
    ax_main.set_title('Refractive Index of ThorLabs Lens Materials in Terms of Wavelength (nm)')
    ax_main.legend(loc = 'lower left')
    ax_main.grid(True)

    # Create the inset plot for the full range (200 - 6000 nm)
    ax_inset = fig.add_axes([0.59, 0.545, 0.3, 0.3])  # [left, bottom, width, height]
    ax_inset2 = fig.add_axes([0.59, 0.178, 0.3, 0.3])
    # Plot the full range in the inset
    ax_inset.plot(wavelengths_full, refractive_indices_full, color='red')
    ax_inset.plot(wavelengths_full, refractive_indices_full_CaF2, color='blue')
    ax_inset2.plot(wavelengths_zoomed, refractive_indices_zoomed_Air, color='green')
    # Labels and title for the inset plot
    ax_inset.set_xlabel('Wavelength (nm)')
    ax_inset.set_ylabel('Refractive Index (n)')
    ax_inset.set_title('Full Range')
    ax_inset.grid(True)
    ax_inset2.grid(True)

    # Show the plot
    plt.show()
    return

if __name__ == "__main__":
    # Example usage
    plot_refractive_index()
    wavelength = 262e-9
    bandwidth = 2e-9
    wavelengthBand = np.linspace(wavelength - bandwidth/2,
                           wavelength + bandwidth/2, 10)
    print(FusedSilica(wavelength))


