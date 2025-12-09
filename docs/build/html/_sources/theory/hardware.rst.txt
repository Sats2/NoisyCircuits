Hardware Considerations
========================

The end goal of NoisyCircuits is to use realistic noise models derived from actual quantum hardware to simulate quantum circuits. Naturally, the next step is to run these circuits on real quantum hardware. This section discusses the hardware considerations and configurations necessary for effectively utilizing NoisyCircuits with IBM quantum devices (but can be extended to other backends by users).


Basis Gates
------------
Every hardware with a Quantum Processing Unit (QPU) has a specific set of basis gates that it natively supports. When designing quantum circuits, it is essential to ensure that the gates used in the circuit are compatible with the target hardware's basis gates. In cases where the circuit contains gates not supported by the hardware, a process called gate decomposition is employed. Gate decomposition breaks down complex gates into sequences of simpler gates that are part of the hardware's basis set. This decomposition is a crucial step to running circuits on real quantum devices, but the process of decomposition itself is not so straightforward and introduces additional errors due to the increased number of gates. 

There are various strategies for gate decomposition, and the choice of strategy can significantly impact the overall performance and fidelity of the quantum circuit when executed on hardware. Additionally, a simple gate decomposition for a single qubit gate may become very complex when extending the gate with a control operation. An example of this can be the Phase Gate which can be decomposed into a :math:`R_z` rotation gate with an associated global phase as shown below:

.. math::
    \begin{align*}
    P(\theta) &= \begin{pmatrix}1 & 0 \\
    0 & e^{i\theta}
    \end{pmatrix} \\ \\
    &= e^{i\theta/2} R_z(\theta) \\ \\
    &= e^{i\theta/2} \begin{pmatrix}
    e^{-i\theta/2} & 0 \\
    0 & e^{i\theta/2}
    \end{pmatrix}
    \end{align*}

However, when extending the Phase Gate to a controlled phase gate, the decomposition is not the controlled- :math:`R_z` gate but rather a sequence of multiple single and two qubit gates. This is due to the fact that a simple controlled :math:`Z` rotation causes the global phase in the single qubit decomposition to leak in as a relative phase which does have a significant influence on the measurements. This increases the complexity of gate decomposition for any arbitrary quantum gate. 

This package currently supports the IBM Heron :math:`R.X` :cite:p:`ibm_heron` and Eagle :math:`R.X` :cite:p:`ibm_eagle` backends for noise modeling and circuit execution. A handful of single and two qubit gates are supported by the package that are decomposed in terms of the basis gates of these backends. The decomposition of the quantum gates are done in way that the resulting sequence of gates produces the exact unitary operation as the original gate without any global phase factors or other approximations. Hence, the list of supported gates are relatively low but are expanding with time. Users can also implement custom gate decompositions for unsupported gates if needed. To see how users can contribute to decomposition on new backends or gates, please refer to the :doc:`../contributing` section.


Qubit Maps
-----------

For two qubit gates, it is necessary to perform swap operations to connect the qubits before executing the intended gate operation followed by another set of swap operations to restore the original qubit mapping. Additionally, the directionality of two qubit gates must be taken into account as the directionality indicates which qubit can act as the control and which as the target qubit on the actual hardware. NoisyCircuits automatically handles these considerations when simulating circuits with two qubit gates. The package uses the coupling map of the target hardware to determine the optimal sequence of swap operations required to connect the qubits for two qubit gate operations. This ensures that the simulated circuits accurately reflect the constraints and capabilities of the target quantum hardware.

In the figure below, the qubit map of the IBM Eagle QPU from the backend "IBM_Brisbane" (utilizing the FakeBrisbane :cite:p:`ibm_brisbane` as the original hardware is now retired) is shown. The arrows indicate the directionality of the two qubit gates, where the tail of the arrow represents the control qubit and the head represents the target qubit. For instance, a CNOT gate can be applied with qubit 1 as the control and qubit 0 as the target, but not vice versa without additional operations. And in cases where a two qubit operation between qubit 0 and qubit 4 is needed, swap operations must be performed to bring the qubits adjacent to each other in accordance with the coupling map. 

.. figure:: ../assets/coupling_map_brisbane.svg
    :alt: Coupling Map of IBM Brisbane Eagle QPU
    :align: center
    :width: 700px

    Figure 1: Coupling Map of IBM Brisbane Eagle QPU using Qiskit :cite:p:`qiskit`

Consider the simple example where a CNOT gate is to be applied between qubit :math:`0` and qubit :math:`4`. Then the sequence of operations would be as follows:

.. math::
    \text{SWAP}(4,3) \rightarrow \text{SWAP}(3,2)
    \rightarrow \text{SWAP}(2,1) \rightarrow \text{op-CNOT}(1,0)\text{-op}
    \rightarrow \text{SWAP}(2,1) \rightarrow \text{SWAP}(3,2)
    \rightarrow \text{SWAP}(4,3)

.. math::
    \text{SWAP}(4,3) \rightarrow \text{SWAP}(3,2) \rightarrow \text{SWAP}(2,1)

.. math::
    \rightarrow \text{op} \rightarrow \text{CNOT}(1,0) \rightarrow \text{op}

.. math::
    \rightarrow \text{SWAP}(2,1) \rightarrow \text{SWAP}(3,2) \rightarrow \text{SWAP}(4,3)

The term :math:`\text{op}\rightarrow\text{CNOT}(1,0)\rightarrow\text{op}` indicates that some additional operations are added to allow for swap of the control and target qubits to match the directionality of the :math:`CNOT` gate as per the coupling map and then swapped back after the gate operation. This is generally done by changing the computational basis of the qubits, performing the two qubit operation and then reverting the basis change. This process also introduces additional errors due to the increased number of gates.

The figure below shows the qubit map of the IBM Heron QPU from the backend "IBM_Fez" :cite:p:`ibm_fez`. Similar to the Eagle QPU, the arrows indicate the directionality of the two qubit gates. But the structure of the coupling map is different from that of the Eagle QPU. As seen from the figure, all two qubit gates are bidirectional in this hardware, meaning that either qubit can act as the control or target qubit without additional operations. This characteristic simplifies the implementation of two qubit gates on this hardware compared to the Eagle QPU. Additionally, the connectivity between qubits is different and with the hexagonal structure, certain qubit pairs are more directly connected than others. This affects the number of swap operations needed for two qubit gates between non-adjacent qubits.

.. figure:: ../assets/gate_map_heron.svg
    :alt: Gate Map of IBM Fez Heron QPU
    :align: center
    :width: 700px

    Figure 2: Gate Map of IBM Fez Heron QPU using Qiskit :cite:p:`qiskit`