# Phase-Plate-Modeling

**Description:**  
This project simulates phase plates and their capabilities in shaping transverse UV laser beams. It provides tools and models to analyze and design phase plates for various applications.

**Table of Contents:**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- Simulation of phase plates for UV laser beam shaping.
- Implementation of algorithms such as Iterative Fourier Transform Algorithm (IFTA) and Fresnel Gaussian Simulation.
- Tools for visualizing and analyzing phase plate designs.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/thomaspjc/Phase-Plate-Modeling.git
   cd Phase-Plate-Modeling
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Dependencies:**  
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Example 1 : Creating a Truncated Gaussian Beam Phase & Simulating its Propagation**

```python
from FresnelGSA import Gaussian, Propagate, Lens, phasePlate

# --- Initialize the simulation with parameters ---
wavelength = 253 * 1e-9 # Setting the wavelength of the beam
w0 = 8 * 1e-3 # Setting the intial beam width
f = 1.2 # Setting the focal length of the lens
extent = [-1.27 * 1e-2, 1.27 * 1e-2] # Setting the plots extent
z0 = pi/wavelength * w0**2 # Determine the complex beam parameter
savefile = 'IFTAPhases/TestRun.h5' # Save file for the Simulated Data
hologramSave = 'IFTAPhases/TestRun.h5' # Save file for the phase mask
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
                 save = 'Phase8.h5', f = f, randomSeed = 15)# [30,target]

# --- Applying a lens transformation to the beam ---
lens = Lens(plate, f)

# --- Propagating the beam to the fourier plane --- 
prop = Propagate(lens, f, plot = True, padding = 0, gaussianProp = False, save = 'SimulatedData8.h5')
```

**Example 2 : Simulating the Transport through a saved Phase Mask**
```python
from FresnelGSA import Gaussian, Propagate, Lens, phasePlate


# --- Initialize the simulation with parameters ---
wavelength = 253 * 1e-9 # Setting the wavelength of the beam
w0 = 8 * 1e-3 # Setting the intial beam width
f = 1.2 # Setting the focal length of the lens
extent = [-1.27 * 1e-2, 1.27 * 1e-2] # Setting the plots extent
z0 = pi/wavelength * w0**2 # Determine the complex beam parameter
savefile = 'IFTAPhases/TestRun.h5' # Save file for the Simulated Data
hologramSave = 'IFTAPhases/TestRun.h5' # Save file for the phase mask
randomSeed = 15
np.random.seed(randomSeed) #Setting the random seed for the IFTA
```

**Example 3 : Testing the accuracy of the Fresnel Propagator through Gaussian Optics**

## Project Structure

- `FresnelGSA.py`: The Main file for Phase Plate Transport Simulation
- `IFTA.py`: Contains the Optical Iterative Fourier Transform Algorithm for phase retrieval.
- `ToolsIFTA.py`: Utility Functions for the Optical IFTA
- `PhysicalPlate.py`: Adapts the Phase Plate model for Manufacturing Standards
- `PlottingTools.py`: Utility functions for plotting and visualizing results.
- `padding.py` : Utility functions to add padding for more accurate Fourier Transforms
- `Targets/`: Directory containing target patterns
- `PlottingTools.py`: Contains the Stanford Colormap for Plotting
- `README.md`: Project overview and instructions.




## License

