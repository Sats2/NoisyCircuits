Method Comparisons
===================
This section provides a comparison between different methods for simulating noisy circuits. The focus is on two main aspects of the software. The first is the ability to filter out noise from the calibration data. The second is the performance comparison between Monte Carlo Wave Function (MCWF) simulations and Density Matrix simulations. 

Metrics
--------
To compare the different methods, we use the following metrics:

- **Battacharyya Coefficient:**  
  The Battacharyya Coefficient (BC) is a measure of similarity between two probability distributions.  
  It is defined as shown below, where :math:`P` and :math:`Q` are the two distributions being compared. 
  A BC value of :math:`1` indicates identical distributions, while a value of :math:`0` indicates completely disjoint distributions. :cite:p:`BC_BD`

  .. math::

     BC(P, Q) = \sum_i \sqrt{P(i)\, Q(i)}

- **Battacharyya Distance:**  
  The Battacharyya Distance (BD) is derived from the Battacharyya Coefficient and is defined as shown below. 
  A BD value of :math:`0` indicates identical distributions, while larger values indicate greater dissimilarity. :cite:p:`BC_BD`  

  .. math::

     BD(P, Q) = -\ln(\sum_i \sqrt{P(i)\, Q(i)})

- **Hellinger Distance:**  
  The Hellinger Distance (HD) is another measure of similarity between two probability distributions. It is defined as shown below. 
  A HD value of :math:`0` indicates identical distributions, while a value of :math:`1` indicates completely disjoint distributions. :cite:p:`HD`

  .. math::

     HD(P, Q) =  \frac{1}{\sqrt{2}} \sqrt{\sum_i (\sqrt{P(i)} - \sqrt{Q(i)})^2}

- **Jensen-Shannon Divergence:**  
  The Jensen-Shannon Divergence (JSD) is a symmetric measure of similarity between two probability distributions. It is defined as shown below, where :math:`M = \frac{1}{2}(P + Q)` and :math:`D_{KL}` is the Kullback-Leibler divergence. 
  A BD value of :math:`0` indicates identical distributions, while larger values indicate greater dissimilarity. :cite:p:`JS`

  .. math::

     JSD(P, Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)



Setup
------

For both comparisons, we use a randomized quantum circuit. The circuit consists of layers of single qubit gates followed by entangling gates. The depth of the circuit is varied to observe the performance under different noise levels. Only single qubit gates are subject to noise and the entangling gates are assumed to be ideal. The randomized circuit blueprint is shown in Figure 1.

.. figure:: ../assets/randomized_circuit.png
    :alt: Randomized Circuit Blueprint
    :align: center
    :width: 60%

    Figure 1: Randomized Circuit Blueprint

The Randomized Circuit generation for the two cases are as follows:

- **Noise Filtering:** The circuit size is fixed to :math:`4` qubits. The single qubit gates are chosen randomly from :math:`\{X, Y, Z, \sqrt{X}, H, R_x(\theta), R_y(\theta), R_z(\theta)\}` and the entagling gates are chosen randomly from :math:`\{CX, CY, CZ\}`. The rotation angles :math:`\theta` is sampled from a normal distribution :math:`\mathcal{N}(0, 2\pi)`. The initial state of the circuit is also randomly initialied from uniform distribution over the Bloch sphere, but is the same for all depths. This randomized circuit setup is repeated :math:`100` times for each of the depths :math:`\{1, 5, 10, 20, 50, 100, 250\}`, with each repeated run starting at the different initial state. The noise model used for each single qubit gate is given below:

+-------------------+-----------------------+
| Error Instruction |      Probability      |
+-------------------+-----------------------+
|        I-I        |   0.9521977439010979  |
+-------------------+-----------------------+
|        I-Z        |   0.0164082737956068  |
+-------------------+-----------------------+
|      I-Reset      |  0.013588129921428271 |
+-------------------+-----------------------+
|        X-I        | 1.777030358953864e-05 |
+-------------------+-----------------------+
|        X-Z        | 2.939666582579276e-10 |
+-------------------+-----------------------+
|      X-Reset      | 2.434416438154702e-09 |
+-------------------+-----------------------+
|        Y-I        |  0.01777030358953864  |
+-------------------+-----------------------+
|        Y-Z        | 2.939666582579276e-10 |
+-------------------+-----------------------+
|      Y-Reset      | 2.434416438154702e-09 |
+-------------------+-----------------------+
|        Z-I        | 1.777030358953864e-05 |
+-------------------+-----------------------+
|        Z-Z        | 2.939666582579276e-10 |
+-------------------+-----------------------+
|      Z-Reset      | 2.434416438154702e-09 |
+-------------------+-----------------------+

- **MCWF Vs Density Matrix Simulation:** The circuit size is varied from :math:`2` qubits to :math:`9` qubits and the depth is varied from :math:`1` to :math:`200` layers. The single qubit gates are chosen randomly from :math:`\{X, \sqrt{X}\}` and the entangling gate is fixed to :math:`ECR`, which are the basis gates of the IBM Eagle QPU. The noise model used is generated from the calibration data of the IBM Eagle device and applied to both single qubit and two qubit gates. The initial state of the circuit is fixed to :math:`|0\rangle^{\otimes n}` for all runs. For this case, the results with the Battacharyya Coefficient and the Jensen-Shannon Divergence are presented for depths of :math:`\{1, 50, 200\}`. For the MCWF simulations, a total of :math:`50` cores were used to parallelize the trajectory simulations. The number of trajectories used are varied from :math:`10` to :math:`1000` to observe the influence on the accuracy of the results.


Noise Filtering
----------------
Noise models from real-world quantum devices show a relatively large chance of success, especially for single qubit gates :cite:p:`GEZ21`. The noise model built from the calibration data generated from the `Randomized Benchmark <https://en.wikipedia.org/wiki/Randomized_benchmarking>`_ tests :cite:p:`randomized_benchmark` performed on the hardware is very comprehensive. The idea of noise filtering is to remove low probability noise instructions from the noise model to reduce the computational overhead from many Kraus operators with very low probabilities and reduce the memory consumption during the simulation.

.. figure:: ../assets/noise_cutoff.png
    :alt: Noise Filtering Results
    :align: center
    :width: 90%

    Figure 2: Noise Filtering Results

From figure 2, it can be inferred that the noise model can be safely filtered as long as the specified threshold is low enough to not significantly affect the open quantum system evolution.


MCWF Vs Density Matrix Simulation
----------------------------------

In this section, we compare the performance of the Monte Carlo Wave Function (MCWF) simulation method against the Density Matrix simulation method for varying circuit sizes and depths. The comparison is based on the Battacharyya Coefficient and the Jensen-Shannon Divergence metrics. For the MCWF simulations, we also compare the influence of different numbers of trajectories on the accuracy of the results to determine a reasonlable number of trajectories to use.

.. container:: plot-grid

   .. container:: plot-row

      .. image:: ../assets/BC_depth_1.svg
         :width: 45%
         :alt: BC comparison at depth 1
         :class: plot-img

      .. image:: ../assets/BC_depth_50.svg
         :width: 45%
         :alt: BC comparison at depth 50
         :class: plot-img

   .. container:: plot-row

      .. image:: ../assets/BC_depth_200.svg
         :width: 45%
         :alt: BC comparison at depth 200
         :class: plot-img

   .. container:: plot-caption

      Figure 3: Battacharyya Coefficient Comparison between MCWF and Density Matrix Simulations at depths 1, 50, and 200.

From figure 3, it can be observed that the MCWF simulation method achieves a high Battacharyya Coefficient compared to the Density Matrix simulation method across all circuit sizes and depths. The accuracy of the MCWF method improves with an increasing number of trajectories, with :math:`1000` trajectories providing results that are very close to those of the Density Matrix method. This behaviour is consistant across the different depths tested. This is also reflected in the Jensen-Shannon Divergence results shown in figure 4.

.. container:: plot-grid

   .. container:: plot-row

      .. image:: ../assets/JS_Depth_1.svg
         :width: 45%
         :alt: JS comparison at depth 1
         :class: plot-img

      .. image:: ../assets/JS_Depth_50.svg
         :width: 45%
         :alt: JS comparison at depth 50
         :class: plot-img

   .. container:: plot-row

      .. image:: ../assets/JS_Depth_200.svg
         :width: 45%
         :alt: JS comparison at depth 200
         :class: plot-img

   .. container:: plot-caption

      Figure 4: JS Divergence Comparison between MCWF and Density Matrix Simulations at depths 1, 50, and 200.

From both figure 3 and 4, it can be concluded that the MCWF simulation method is a viable alternative to the Density Matrix simulation method, especially for larger circuits where the computational resources required for Density Matrix simulations become prohibitive. The choice of the number of trajectories in the MCWF method is crucial for achieving a balance between accuracy and computational efficiency. From the above conducted experiments, using approximately :math:`100` trajectories seems to provide a good trade-off for most scenarios, but more trajectories may be required at higher circuit depths for a larger qubit system size to maintain accuracy.

Additionally, we also compare the execution time of both simulation methods as well as the memory consumption during the simulations. The results are summarized in figure 5 and 6 below.

.. container:: plot-grid

   .. container:: plot-row

      .. image:: ../assets/Runtime_Comparison_Depth_1.svg
         :width: 45%
         :alt: Runtime comparison at depth 1
         :class: plot-img

      .. image:: ../assets/Runtime_Comparison_Depth_50.svg
         :width: 45%
         :alt: Runtime comparison at depth 50
         :class: plot-img

   .. container:: plot-row

      .. image:: ../assets/Runtime_Comparison_Depth_200.svg
         :width: 45%
         :alt: Runtime comparison at depth 200
         :class: plot-img

   .. container:: plot-caption

      Figure 5: Runtime Comparison between MCWF and Density Matrix Simulations at depths 1, 50, and 200.


.. container:: plot-grid

   .. container:: plot-row

      .. image:: ../assets/Memory_Comparison_Depth_1.svg
         :width: 45%
         :alt: Memory comparison at depth 1
         :class: plot-img

      .. image:: ../assets/Memory_Comparison_Depth_50.svg
         :width: 45%
         :alt: Memory comparison at depth 50
         :class: plot-img

   .. container:: plot-row

      .. image:: ../assets/Memory_Comparison_Depth_200.svg
         :width: 45%
         :alt: Memory comparison at depth 200
         :class: plot-img

   .. container:: plot-caption

      Figure 6: Memory Comparison between MCWF and Density Matrix Simulations at depths 1, 50, and 200.


Conclusion
----------
From the comparisons conducted, it is evident that both noise filtering and the MCWF simulation method offer significant advantages in terms of computational efficiency without compromising accuracy. Noise filtering effectively reduces the complexity of the noise model, leading to faster simulations. The MCWF method provides a scalable alternative to Density Matrix simulations, particularly for larger circuits, while still maintaining high accuracy with an appropriate choice of trajectories. These methods are valuable tools for simulating noisy quantum circuits and can be tailored to specific requirements based on the desired balance between accuracy and computational resources.