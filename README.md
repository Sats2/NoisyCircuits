# NoisyCircuits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

A Python package for creating and simulating noisy quantum circuits using realistic error models from IBM Eagle R3 quantum hardware. The package implements the Monte-Carlo Wave Function (MCWF) method for efficient statevector simulation of noisy quantum systems.

## 🎯 Overview

NoisyCircuits enables researchers and developers to:

- **Simulate realistic quantum noise** using calibration data from IBM Eagle R3 chipsets
- **Perform efficient noisy simulations** with the Monte-Carlo Wave Function method
- **Validate quantum algorithms** under realistic hardware conditions  
- **Develop noise-aware quantum machine learning** applications
- **Compare quantum algorithms** between ideal and noisy regimes

### Key Features

✨ **Hardware-Calibrated Noise Models**: Direct integration with IBM Quantum backend calibration data  
🚀 **Parallel Monte-Carlo Simulation**: Multi-core trajectory execution for scalable performance  
🎛️ **Flexible Gate Set**: Support for IBM Eagle R3 basis gates (X, √X, Rz, ECR)  
📊 **Validation Framework**: Built-in comparison with exact density matrix methods  
🔬 **Research Applications**: Ready-to-use examples for quantum machine learning and algorithm development  

### Supported Quantum Gates

- **Single-qubit gates**: X, √X, Rz(θ), Hadamard, Y, Z, S, T, and parameterized rotations
- **Two-qubit gates**: ECR (Echoed Cross-Resonance), CNOT, CZ, and controlled operations
- **Multi-qubit gates**: Toffoli and other controlled operations

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- A conda-based environment manager ([Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Anaconda](https://www.anaconda.com/), or [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html))
- IBM Quantum account and API token (for noise model access)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sats2/NoisyCircuits.git
   cd NoisyCircuits
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate NoisyCircuits
   ```

3. **Install the package:**
   ```bash
   pip install .
   ```

### Alternative Installation (Development Mode)

For development or if you plan to modify the code:
```bash
pip install -e .
```

### Dependencies

Core dependencies are automatically installed:
- **PennyLane**: Quantum computing framework
- **Qiskit**: IBM Quantum software stack
- **Ray**: Distributed computing for parallel execution
- **NumPy, Matplotlib**: Scientific computing and visualization

## 📁 Repository Structure

```
NoisyCircuits/
├── src/NoisyCircuits/          # Main package source code
│   ├── __init__.py
│   ├── QuantumCircuit.py       # Core quantum circuit class
│   └── utils/                  # Utility modules
│       ├── GetNoiseModel.py    # IBM backend integration
│       ├── BuildQubitGateModel.py
│       ├── DensityMatrixSolver.py
│       ├── PureStateSolver.py
│       └── ParallelExecutor.py
├── test/                       # Test suite and examples
│   ├── README.md              # Detailed test documentation
│   ├── introduction.ipynb     # Getting started tutorial
│   ├── method_verification.ipynb  # Validation against exact methods
│   ├── quantum_neural_networks.ipynb  # ML application example
│   └── design_study_single_feature.csv  # Sample dataset
├── environment.yml            # Conda environment specification
├── setup.py                  # Package installation configuration
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

## 🧪 Testing and Validation

The `test/` directory contains comprehensive validation and example notebooks:

### Validation Framework
- **Method Verification**: Statistical comparison between MCWF and exact density matrix simulation
- **Hardware Integration**: Validation using real IBM Quantum backend data
- **Performance Benchmarking**: Trajectory convergence and computational efficiency analysis

### Example Applications
- **Quantum Machine Learning**: CFD parameter prediction using quantum neural networks
- **Algorithm Comparison**: Performance analysis under realistic noise conditions
- **Educational Resources**: Step-by-step tutorials for quantum noise simulation

**Quick Test**: Run the method verification notebook to validate your installation:
```bash
jupyter notebook test/method_verification.ipynb
```

For detailed information about the test suite, see [`test/README.md`](test/README.md).

## 📚 Documentation and Examples

### Tutorials
1. **[Introduction](test/introduction.ipynb)**: Basic usage and configuration
2. **[Method Verification](test/method_verification.ipynb)**: Validation and accuracy testing  
3. **[Quantum Neural Networks](test/quantum_neural_networks.ipynb)**: Machine learning applications

### Key Concepts
- **Monte-Carlo Wave Function**: Efficient method for simulating open quantum systems
- **Hardware Noise Models**: Using real device calibration data for realistic simulations
- **Parallel Execution**: Scaling simulations across multiple CPU cores
- **Statistical Validation**: Ensuring simulation accuracy through multiple metrics

## 🤝 Contributing

We welcome contributions to NoisyCircuits! Here's how you can help:

### Types of Contributions

- 🐛 **Bug Reports**: Report issues or unexpected behavior
- ✨ **Feature Requests**: Suggest new functionality or improvements
- 📝 **Documentation**: Improve tutorials, examples, or API documentation
- 🧪 **Testing**: Add test cases or improve validation coverage
- 💻 **Code Contributions**: Implement new features or optimize existing code

### Development Workflow

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Set up development environment**:
   ```bash
   conda env create -f environment.yml
   conda activate NoisyCircuits
   pip install -e .  # Install in development mode
   ```

3. **Make your changes** and add tests if applicable

4. **Run the validation suite**:
   ```bash
   jupyter notebook test/method_verification.ipynb
   ```

5. **Submit a pull request** with a clear description of your changes

### Contribution Guidelines

- **Code Style**: Follow PEP 8 python style guidelines
- **Documentation**: Update docstrings and README files for new features
- **Testing**: Include test cases for new functionality
- **Backwards Compatibility**: Maintain compatibility with existing APIs when possible

### Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Start a GitHub Discussion for questions and ideas

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support and Contact

- **Author**: Sathyamurthy Hegde
- **GitHub**: [@Sats2](https://github.com/Sats2)

## 🙏 Acknowledgments

- **IBM Quantum** for providing access to quantum hardware and calibration data
- **PennyLane** team for the excellent quantum computing framework
- **Qiskit** community for quantum software development tools
- **Research Community** for feedback and contributions to quantum noise modeling

## 📊 Performance Metrics

Based on validation testing:
- **Accuracy**: >99.99% fidelity compared to exact density matrix simulation
- **Scalability**: Efficient parallel execution across multiple CPU cores
- **Convergence**: Consistent results with as few as 100 Monte-Carlo trajectories
- **Hardware Integration**: Seamless integration with IBM Eagle R3 chipset data

---

*For more detailed information, examples, and tutorials, please refer to the documentation in the `test/` directory.*
