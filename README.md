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
from FresnelGSA import FresnelGSA
from PlottingTools import plot_phase_plate

# Initialize the simulation with parameters
simulation = FresnelGSA(parameters)

# Run the simulation
result = simulation.run()

# Visualize the phase plate
plot_phase_plate(result)
```

**Example 2 : Simulating the Transport through a saved Phase Mask**

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

