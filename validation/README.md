# Validation Directory - NoisyCircuits

This directory contains the method validation notebook for the NoisyCircuits library. The NoisyCircuits library enables simulation of quantum circuits with noise models using the Monte-Carlo Wave Function (MCWF) method.

## Contents

### 📓 Jupyter Notebooks

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

### 📊 Data Files

#### `Results_Log_File.txt`
Comprehensive validation results from method verification tests:
- Performance metrics across different trajectory counts (10, 100, 200, 500)
- Circuit depth and qubit count variations
- Statistical distance measurements between MCWF and the density matrix method
- Computational benchmarking data


## Results Summary

Based on validation testing (see `Results_Log_File.txt`):
- **Accuracy:** MCWF method achieves >99.99% fidelity vs. density matrix simulation
- **Statistical Distance:** Typical Battacharyya distances < 3×10⁻⁵
- **Convergence:** Consistent results across 10-500 trajectory runs
- **Scalability:** Validated for circuits up to multiple qubits and variable depth