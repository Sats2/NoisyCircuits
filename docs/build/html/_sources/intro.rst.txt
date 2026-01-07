NoisyCircuits
=============

Create and simulate noisy quantum circuits using IBM (Heron R.X / Eagle R.X) calibration data. NoisyCircuits implements the Monte-Carlo Wave Function (MCWF) method for efficient statevector simulation under realistic noise.

Overview
--------

NoisyCircuits helps you quickly:

- **Simulate realistic noise:** Build error models from IBM QPU calibration data.
- **Run fast statevector sims:** Use MCWF to simulate noisy dynamics efficiently.
- **Compare algorithms:** Evaluate ideal vs. noisy performance side by side.
- **Validate designs:** Test robustness under hardware-like conditions.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- A conda-style environment manager: `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, `Anaconda <https://www.anaconda.com/>`_, or `Micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_.
- An IBM Quantum (IBMQ) account + API token (optional but recommended, only needed to pull live hardware data or submit jobs to real devices). Sample Noise models are provided for testing without an IBMQ account.

Steps
~~~~~

1. Clone the repository:

	.. code-block:: bash

		git clone https://github.com/Sats2/NoisyCircuits.git

2. Change into the project directory:

	.. code-block:: bash

		cd NoisyCircuits

3. Create the Python environment from ``environment.yml`` (replace ``conda`` with your manager if needed):

	.. code-block:: bash

		conda env create -f environment.yml

4. Activate the environment:

	.. code-block:: bash

		conda activate NoisyCircuits

5. Install the package:

	.. code-block:: bash

		pip install .

	.. note::

		A PyPI release is planned; installation instructions will be updated when available.

Citing NoisyCircuits
---------------------
If you use NoisyCircuits in your research, please cite the software as follows:

.. code-block:: text
	
	@software{NoisyCircuits2025,
  		author = {Hegde, Sathyamurthy},
  		title = {NoisyCircuits},
  		year = {2025},
  		url = {https://github.com/Sats2/NoisyCircuits},
  		version = {1.1.1},
	}

Project Status
~~~~~~~~~~~~~~~
NoisyCircuits is in active development. New features and improvements are regularly added. Please check the GitHub repository for the latest updates.

Getting Help
------------

- **Issues:** Report bugs and request features at GitHub Issues. Include clear steps to reproduce for bugs and label feature requests appropriately. Typical triage and response occurs within 1–2 weeks.
- **Discussions:** Ask questions and share ideas in GitHub Discussions.

Links
-----

- `Issues <https://github.com/Sats2/NoisyCircuits/issues>`_
- `Discussions <https://github.com/Sats2/NoisyCircuits/discussions>`_