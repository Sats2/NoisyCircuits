# Example Directory - NoisyCircuits

This directory contains test scripts, validation notebooks, and example applications for the NoisyCircuits library. The NoisyCircuits library enables simulation of quantum circuits with noise models using the Monte-Carlo Wave Function (MCWF) method.

## Contents

### 📓 Jupyter Notebooks

#### `introduction.ipynb`
**High-level introduction to the NoisyCircuits library**

This notebook provides a comprehensive tutorial on how to use the NoisyCircuits library, including:
- Setting up IBM Quantum backend connections
- Configuring noise models from real quantum hardware
- Initializing quantum circuits with noise parameters
- Basic usage examples with the Monte-Carlo Wave Function method

**Key Features Demonstrated:**
- IBM Quantum token authentication
- Backend calibration data integration (IBM Eagle / Heron chipset)
- Noise model configuration and optimization
- Multi-core parallel trajectory execution

#### `quantum_neural_networks.ipynb`
**Quantum Machine Learning application for Computational Fluid Dynamics**

This notebook demonstrates a practical application of noisy quantum circuits for machine learning:
- **Problem:** Predicting lift coefficients for varying angles of attack (0-15°)
- **Data:** 2D CFD simulation results for airfoil analysis
- **Method:** Quantum Neural Network with noise-aware training
- **Features:** Single-feature regression (extensible to multi-feature problems)

**Dependencies:** pandas, scipy, scikit-learn, matplotlib, numpy

#### `run_on_hardware.ipynb`
**Tutorial Notebook on Creating and Submitting Quantum Circuits to Hardware**

This notebook provides a tutorial on using the package to create and submit a quantum circuit to IBM Quantum Hardware. The tutorial includes:
- Setting up IBM API access.
- Creating and Submitting Quantum Circuits to quantum hardware.
- Retreiving results from the quantum hardware.
- Fidelity comparison between simulator and hardware results from output probabilities.

#### `run_multiple_on_hardware.ipynb`
**Tutorial Notebook on Creating and Submitting Multiple Quantum Circuits to Hardware**

This notebook provides a tutorial on using the package to create and submit multiple quantum circuits to IBM Quantum Hardware and retreive the results from the hardware.


### 📊 Data Files

#### `design_study_single_feature.csv`
Computational Fluid Dynamics dataset from 2D NACA 4412 airfoil simulations containing:
- **Columns:** Angle of Attack (AoA), Drag Coefficient, Lift Coefficient
- **Range:** 0-15 degrees angle of attack
- **Purpose:** Training data for quantum neural network regression

## Prerequisites

Before running the notebooks in this directory, ensure you have:

1. **Environment Setup:** Installed the NoisyCircuits library following the main repository instructions
2. **Hardware Requirements:** Sufficient computational resources for parallel MCWF simulations
3. **IBM Quantum Access (optional):** Valid IBM Quantum account and API token. In case API access is not feasible, there are sample noise models from hardware calibration that can be loaded instead.

## Usage Instructions

### Quick Start
1. Begin with `introduction.ipynb` to understand basic library usage
<<<<<<< HEAD
2. Explore `quantum_neural_networks.ipynb` for practical applications
3. Look into `run_on_hardware.ipynb` and `run_multiple_on_hardware.ipynb` for creating and submitting quantum circuits to quantum hardware.
=======
3. Explore `quantum_neural_networks.ipynb` for practical applications
>>>>>>> origin/main

### Configuration Notes
- **IBM Backend:** Currently supports IBM Eagle / Heron chipset, but the Eagle QPU devices have been retired (API access to these devices is no longer possible).
- **Basis Gates:** 
    - Eagle QPU - X, √X, Rz(·), ECR
    - Heron QPU - X, √X, Rz(·), Rx(·), CZ, RZZ(·)
- **Parallel Execution:** Configure `num_cores` based on available system resources
- **Noise Threshold:** Adjust `threshold` parameter for noise filtering (default: 1e-12)

### Performance Considerations
- **Trajectory Count:** Higher trajectory counts improve statistical accuracy but increase computation time
- **Circuit Depth:** Deeper circuits require more trajectories for convergence
- **Qubit Count:** Memory and time complexity scale exponentially with system size

## Example Applications

The notebooks demonstrate three key use cases:
1. **Educational:** Learning quantum noise simulation fundamentals
3. **Research:** Applying noisy quantum computing to real-world problems

## Support and Documentation

For additional information:
- Main repository documentation: [NoisyCircuits README](../README.md)
- IBM Quantum documentation: [IBM Quantum Cloud Setup](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup)