Tutorials
==========

Overview
--------

This section contains tutorials to help you get started with NoisyCircuits. The tutorials cover a range of topics, from basic usage to more advanced features, and are designed to help you understand how to use the package effectively. Each tutorial is presented as a Jupyter notebook, allowing you to follow along with the code and see the results in real-time.

The tutorials include:

- **Introduction to NoisyCircuits:** A beginner-friendly introduction to the package, covering installation, basic concepts, and simple examples.
- **Running Circuits on Hardware:** A guide to executing quantum circuits on real IBM quantum hardware using NoisyCircuits.
- **Running Multiple Circuits on Hardware:** Instructions on how to run multiple quantum circuits on IBM quantum hardware and analyze the results.
- **Quantum Neural Networks:** An exploration of how to implement and simulate quantum neural networks using NoisyCircuits. The notebook looks into building and training QNNs with realistic noise models for the prediction of the lift co-efficient for the NACA 4412 airfoil. The data for training and testing is available within the repository. The data is obtained from STAR-CCM+ simulations.

Each tutorial is designed to be self-contained, so you can start with any of them based on your interests and needs. We recommend starting with the introduction tutorial if you are new to NoisyCircuits.

Prerequisites
-------------
Before starting the tutorials, ensure you have the following prerequisites:

- A working installation of NoisyCircuits (see the Installation section in :doc:`intro`).
- Sufficient compute resources to run quantum circuit simulations (if running in parallel, it is recommended to use at most 75% of available cores).
- An IBM Quantum account and API token (only needed for tutorials involving real hardware, or in cases where noise data from machine calibration is needed.).

Configuration
--------------
- **IBM Backend:** The package currently supports the IBM Heron RX and Eagle RX backends for noise modeling. Ensure you have access to these backends via your IBM Quantum account (but as of 1st November 2025, the Eagle RX QPU is deprecated and API access is no longer possible).
- **Parallel Execution:** Configure the `num_cores` parameter based on your system's capabilities to optimize simulation performance.
- **Noise Threshold:** Adjust the noise threshold settings with `threshold` parameter to filter out low probability noise events during simulations. This speeds up simulations without significantly affecting accuracy as long as only very low probability events are filtered. `Default: 1e-12`, not recommended to set the threshold higher than `1e-4`.

.. toctree::
    :maxdepth: 1
    :caption: Tutorials List (Jupyter Notebooks):

    examples/introduction
    examples/run_on_hardware
    examples/run_multiple_on_hardware
    examples/quantum_neural_networks

Downloads
----------

All the tutorial notebooks along with the sample noise models can be downloaded from here by clicking the link below.

:download:`Download Tutorials <assets/tutorials.zip>`