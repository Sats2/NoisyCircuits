# CHANGE LOG

All notable changes to this project is documentated here based on the format by [Keep a Changelog](http://keepachangelog.com/). This project adheres to [Semantic Versioning](http://semver.org/).

## [v1.3.0] - 2026-03-10

### Added Functionlity

* Added functionality to get the raw noise model from the calibration data that can be provided as CSV files to allow for users to use their own backends (see documentation how to use custom backends).

### Updates

* Modified functionality to get the raw noise model from the calibration data of IBM backends via API to replace the existing module.
* Modified the density matrix and execute methods with qiskit and qulacs solvers for operator casting to little endian representation.
* Modified the readout error information retreival where the average error is now discarded and the specific errors are used.

### Bug Fixes

* Fixed bugs related to noise model decomposition.
* Fixed bug in the density matrix solver of Qulacs.
* Fixed bug related to endian representation in the noise model generation.


## [v1.2.0] - 2026-02-17

### Issue Fixes

* Switched to memory efficient representation for the noise operators.
* Now runs smoothly in a quick and memory efficient manner for upto $20$ qubits due to memory optimization and code optimization for maximum performance from single core runs plus parallel processing of information.

### Added Functionality

* Added additional backends to simulate the quantum circuits. 
* Available backends are Qulacs, Qiskit and Pennylane. Defaults to Qulacs.
* Changed reduced qubit set probability computation from partial trace to marginal probability method.

### Bug Fixes

* Fixed minor bugs that caused issues during noise model generation.
* Fixed bug that causes namespace errors when switching between different backend solvers.
* Fixed bug with the endian-ness of the multi-qubit operators during noise model generation along with matrix size definitions.


## [v1.1.1] - 2026-01-06

### Issue Fix

Fixed issue where the noise model causes a memory overflow issue at the $10$ qubit count and requiring a substantially large runtime to build the noise model.

## [v1.1.0] - 2025-12-09

### Updates - QPU Extension and Modularization

* Revamped code base to allow abstraction and modularization for a simplified workflow to update QPU designs and basis gate decomposition.
* Added the Heron Architecture to the code base. Now, the package can be used to simulate circuits decomposed into the basis gates of both IBM Eagle and Heron QPUs. 
* Additionally, users can create their own decomposition and noise models to utilize the package.
* Separated the gate decomposition, swap sequencing and simulator modules from the main Quantum Circuit module.
* Added module to allow users to submit quantum circuits to IBM hardware via API.
* Added unit tests and integration tests.
* Built documentation for the software.


## [v1.0.0] - 2025-08-19

### Package Release

A Python package for creating and simulating noisy quantum circuits using error models from IBM Eagle R3 quantum hardware calibration data fully released.

NoisyCircuits enables researchers and developers to:
* Simulate realistic quantum noise using calibration data from IBM Eagle R3 chipsets
* Perform efficient noisy statevector simulations with the Monte-Carlo Wave Function method
* Validate quantum algorithms under realistic hardware conditions
* Develop noise-aware quantum machine learning applications
* Compare quantum algorithms between ideal and noisy regimes

Supported Quantum Gates:
* Single Qubit Gates: $X$, $Y$, $Z$, $\sqrt{X}$, $H$, $R_x(\theta)$, $R_y(\theta)$ and $R_z(\theta)$.
* Two Qubit Gates: $CX$, $CY$, $CZ$, $ECR$, $SWAP$, $CR_x(\theta)$, $CR_y(\theta)$, $CR_z(\theta)$.
* $N$-Qubit Unitary (but without decomposition and applied as a pure and ideal gate).

Adheres to the qubit layout of the hardware and selects the shortest possible route to connect two different physical qubits on the QPU.