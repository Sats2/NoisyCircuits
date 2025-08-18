# Test Directory - NoisyCircuits

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
- Backend calibration data integration (IBM Eagle R3 chipset)
- Noise model configuration and optimization
- Multi-core parallel trajectory execution

#### `method_verification.ipynb`
**Validation of the MCWF method against density matrix simulations**

This notebook validates the Monte-Carlo Wave Function implementation by comparing probability distributions from:
- MCWF method (statevector simulation with noise)
- Density matrix simulation (exact noisy simulation)

**Validation Metrics:**
- Battacharyya Coefficient and Distance
- Hellinger Distance  
- Jensen-Shannon Divergence

The validation ensures that the MCWF method provides statistically equivalent results to exact density matrix simulations while being computationally more efficient for large systems.

#### `quantum_neural_networks.ipynb`
**Quantum Machine Learning application for Computational Fluid Dynamics**

This notebook demonstrates a practical application of noisy quantum circuits for machine learning:
- **Problem:** Predicting lift coefficients for varying angles of attack (0-15°)
- **Data:** 2D CFD simulation results for airfoil analysis
- **Method:** Quantum Neural Network with noise-aware training
- **Features:** Single-feature regression (extensible to multi-feature problems)

**Dependencies:** pandas, scipy, scikit-learn, matplotlib, numpy

### 📊 Data Files

#### `design_study_single_feature.csv`
Computational Fluid Dynamics dataset containing:
- **Columns:** Angle of Attack (AoA), Drag Coefficient, Lift Coefficient
- **Range:** 0-15 degrees angle of attack
- **Purpose:** Training data for quantum neural network regression

#### `Results_Log_File.txt`
Comprehensive validation results from method verification tests:
- Performance metrics across different trajectory counts (10, 100, 200, 500)
- Circuit depth and qubit count variations
- Statistical distance measurements between MCWF and exact methods
- Computational benchmarking data

## Prerequisites

Before running the notebooks in this directory, ensure you have:

1. **Environment Setup:** Installed the NoisyCircuits library following the main repository instructions
2. **IBM Quantum Access:** Valid IBM Quantum account and API token
3. **Hardware Requirements:** Sufficient computational resources for parallel MCWF simulations

## Usage Instructions

### Quick Start
1. Begin with `introduction.ipynb` to understand basic library usage
2. Run `method_verification.ipynb` to validate your installation
3. Explore `quantum_neural_networks.ipynb` for practical applications

### Configuration Notes
- **IBM Backend:** Currently supports IBM Eagle R3 chipset (e.g., "ibm_brisbane")
- **Basis Gates:** X, √X, Rz(·), ECR
- **Parallel Execution:** Configure `num_cores` based on available system resources
- **Noise Threshold:** Adjust `threshold` parameter for noise filtering (default: 1e-4)

### Performance Considerations
- **Trajectory Count:** Higher trajectory counts improve statistical accuracy but increase computation time
- **Circuit Depth:** Deeper circuits require more trajectories for convergence
- **Qubit Count:** Memory and time complexity scale exponentially with system size

## Example Applications

The notebooks demonstrate three key use cases:
1. **Educational:** Learning quantum noise simulation fundamentals
2. **Validation:** Verifying simulation accuracy against theoretical benchmarks  
3. **Research:** Applying noisy quantum computing to real-world problems

## Support and Documentation

For additional information:
- Main repository documentation: [NoisyCircuits README](../README.md)
- IBM Quantum documentation: [IBM Quantum Cloud Setup](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup)
- Qiskit noise model documentation for advanced customization

## Results Summary

Based on validation testing (see `Results_Log_File.txt`):
- **Accuracy:** MCWF method achieves >99.99% fidelity vs. exact density matrix simulation
- **Statistical Distance:** Typical Battacharyya distances < 3×10⁻⁵
- **Convergence:** Consistent results across 10-500 trajectory runs
- **Scalability:** Validated for circuits up to multiple qubits and variable depth