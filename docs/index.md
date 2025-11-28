# NoisyCircuits

Create and simulate noisy quantum circuits using IBM (Heron RX / Eagle RX) calibration data. NoisyCircuits implements the Monte-Carlo Wave Function (MCWF) method for efficient statevector simulation under realistic noise.

## Overview

NoisyCircuits helps you quickly:

- **Simulate realistic noise:** Build error models from IBM QPU calibration data.
- **Run fast statevector sims:** Use MCWF to simulate noisy dynamics efficiently.
- **Compare algorithms:** Evaluate ideal vs. noisy performance side by side.
- **Validate designs:** Test robustness under hardware-like conditions.

## Installation

### Prerequisites

- A conda-style environment manager: [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Anaconda](https://www.anaconda.com/), or [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).
- An IBM Quantum account + API token (only needed to pull live hardware data or submit jobs to real devices).

### Steps

1) Clone the repository:

```bash
git clone https://github.com/Sats2/NoisyCircuits.git
```

2) Change into the project directory:

```bash
cd NoisyCircuits
```

3) Create the Python environment from `environment.yml` (replace `conda` with your manager if needed):

```bash
conda env create -f environment.yml
```

4) Activate the environment:

```bash
conda activate NoisyCircuits
```

5) Install the package:

```bash
pip install .
```

> Note: A PyPI release is planned; installation instructions will be updated when available.

## Getting Help

- **Issues:** Report bugs and request features at GitHub Issues. Include clear steps to reproduce for bugs and label feature requests appropriately. Typical triage and response occurs within 1–2 weeks.
- **Discussions:** Ask questions and share ideas in GitHub Discussions.

Links:

- [Issues](https://github.com/Sats2/NoisyCircuits/issues)
- [Discussions](https://github.com/Sats2/NoisyCircuits/discussions)