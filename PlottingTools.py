#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:12:02 2025

Stanford ColorMap and other Plotting tools

@author: thomas
"""
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

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