Quantum Computing 101
======================

Quantum Computers
------------------------
Quantum computers are devices that leverage the principles of quantum mechanical phenomena such as superposition and entanglement to perform calculations faster than a classical computer. Potential applications of quantum computing are in the fields of cybersecurity, data analytics and machine learning, and simulation and optimization programs :cite:p:`LS22`.

Qubits 
--------
Analogous to classical bits, quantum computers use quantum bits or qubits as the basic unit of information. Unlike classical bits that can be in one of two states (0 or 1), qubits can exist in a superposition of both states simultaneously. This property allows quantum computers to process a vast amount of information in parallel. Mathematically, a single qubit can be represented as a two-dimensional complex vector in the Hilbert space:

.. math::
   |\psi\rangle = \alpha|0\rangle + \beta|1\rangle

where :math:`|0\rangle` and :math:`|1\rangle` are the basis states, and :math:`\alpha` and :math:`\beta` are complex coefficients such that :math:`|\alpha|^2 + |\beta|^2 = 1` (called the "normalization constraint"). :cite:p:`SML24` Geometrically, a qubit can be visualized on the Bloch sphere, where any point on the sphere represents a valid qubit state. A Bloch sphere is a unit sphere in a three-dimensional space where the north and south poles correspond to the basis states :math:`|0\rangle` and :math:`|1\rangle`, respectively. An example of a qubit state on the Bloch sphere is shown in Figure 1.

.. figure:: ../assets/Bloch_sphere.svg
   :alt: Bloch Sphere
   :width: 300px
   :align: center

   Figure 1: Bloch Sphere, by Smite-Meister, licensed under CC BY-SA 3.0.(Wikimedia Commons), :cite:p:`Smind`

A qubit can theoretically store an infinite amount of information due to its continuous state space. However, when measured, a qubit collapses to one of the basis states, yielding a classical bit of information (0 or 1) with probabilities determined by the coefficients :math:`\alpha` and :math:`\beta`. :cite:p:`NC10` The qubit can Mathematically be represented by:

.. math::
    |\psi\rangle = e^{i\gamma}(\cos(\frac{\theta}{2})|0\rangle + e^{i\varphi}\sin(\frac{\theta}{2})|1\rangle)

where :math:`\gamma`, :math:`\theta`, and :math:`\varphi` are real numbers that define the global phase and the position on the Bloch sphere. For all practical purposes, the global phase :math:`e^{i\gamma}` can be ignored since it does not affect measurement outcomes. Thus, the state of a qubit can be fully described by the angles :math:`\theta` and :math:`\varphi`, which correspond to its position on the surface of the Bloch sphere by, :cite:p:`NC10`

.. math::
    |\psi\rangle = \cos(\frac{\theta}{2})|0\rangle + e^{i\varphi}\sin(\frac{\theta}{2})|1\rangle


Mulitple Qubits and Entanglement
-----------------------------------

Mutli-Qubit Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a system of multiple qubits, the state of the qubit system is obtained from taking the Kronecker product of the states of the different qubits as shown in the equation below for an :math:: `n` qubit system. If the coefficients for each qubit in the system already satisfy the normalization constraint, then the resulting state will also satisfy the normalization constraint.

.. math::
    |q_0q_1...q_{n-1}\rangle = |q_0\rangle \otimes |q_1\rangle \otimes ... \otimes |q_{n-1}\rangle

where :math:`|q_i\rangle` represents the state of the :math:`i^{th}` qubit in the system. For example, let us consider a two-qubit system where the first qubit is in the state :math:`|q_0\rangle = \alpha_0|0\rangle + \beta_0|1\rangle` and the second qubit is in the state :math:`|q_1\rangle = \alpha_1|0\rangle + \beta_1|1\rangle`. The combined state of the two-qubit system can be expressed as:

.. math::
    |q_0q_1\rangle = |q_0\rangle \otimes |q_1\rangle
    = \begin{pmatrix}
    \alpha_0 \\
    \beta_0
    \end{pmatrix} \otimes \begin{pmatrix}
    \alpha_1 \\
    \beta_1
    \end{pmatrix}
    = \begin{pmatrix}
    \alpha_0 \otimes \begin{pmatrix}
    \alpha_1 \\
    \beta_1
    \end{pmatrix} \\
    \beta_0 \otimes \begin{pmatrix}
    \alpha_1 \\
    \beta_1
    \end{pmatrix}
    \end{pmatrix}
    = \begin{pmatrix}
    \alpha_0\alpha_1 \\
    \alpha_0\beta_1 \\
    \beta_0\alpha_1 \\
    \beta_0\beta_1
    \end{pmatrix}

.. math::
    |q_0q_1\rangle = \alpha_0\alpha_1|00\rangle + \alpha_0\beta_1|01\rangle + \beta_0\alpha_1|10\rangle + \beta_0\beta_1|11\rangle

where :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`, and :math:`|11\rangle` are the basis states of the two-qubit system. :cite:p:`SML24`


Entanglement
^^^^^^^^^^^^^^

Entanglement is a unique quantum phenomenon where the states of two or more qubits become correlated in such a way that the state of one qubit cannot be described independently of the state of the other qubits, regardless of the distance between them. This property is a key resource for many quantum algorithms and protocols. :cite:p:`SML24` For example, consider the Bell state, which is an entangled state of two qubits:

.. math::
    |\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)

In this state, if the first qubit is measured and found to be in the state :math:`|0\rangle`, the second qubit will instantaneously collapse to the state :math:`|0\rangle` as well, and similarly for the state :math:`|1\rangle`. This correlation holds true even if the two qubits are separated by large distances. :cite:p:`SML24`


Qubit Operations
-----------------

Analogous to classical logic gates, quantum gates are the fundamental building blocks of quantum circuits that manipulate qubits. Quantum gates are represented by unitary matrices that operate on the state vectors of qubits. Unlike classical gates, quantum gates are fully reversible. They can be viewed as operators that evolve the state of the quantum system described by the Schrödinger equation. :cite:p:`NC10`

.. math::
    i\hbar\frac{d}{dt}|\psi\rangle = \hat{H}|\psi\rangle

where :math:`\hbar` is the reduced Planck constant, :math:`\hat{H}` is the Hamiltonian operator representing the total energy of the system, and :math:`|\psi\rangle` is the state vector of the quantum system. The solution to this equation describes how the quantum state evolves over time under the influence of the Hamiltonian. Under stable conditions (closed systems), the system has a constant Hamiltonian that describes the state evolution over time. The time evolution operator :math:`U(t)` is given by:

.. math::
    U(t) = e^{-\frac{i\hat{H}t}{\hbar}}

Under steady conditions the quantum gate operation can be represented as:

.. math::
    |\psi'\rangle = U|\psi\rangle

where :math:`|\psi\rangle` is the initial state of the qubit(s), :math:`U` is the unitary matrix representing the quantum gate, and :math:`|\psi'\rangle` is the resulting state after the gate operation. :cite:p:`NC10` Hence, quantum gates are unitary operators (:math:`U^{\dagger}U=I`) that preserve the normalization constraint of qubit states, due to the condition of reversibility in quantum mechanics. The most commonly used single and two qubit gates are available in the NoisyCircuits library and their list can be found in the documentation in :mod:`NoisyCircuits.utils.Decomposition`.

Measurements
-------------

Measurement is the process used to extract classical information from quantum states by collapsing the superposition of qubit states into one of its basis states with a certain probability. Measurement is non-unitary and irreversible. When a qubit is measured, its state collapses to one of the basis states, and the outcome is probabilistic, determined by the coefficients of the superposition prior to measurement. For a single qubit in the state :math:`|\psi\rangle = \alpha|0\rangle + \beta|1\rangle`, measuring the qubit in the computational basis yields the outcome :math:`|0\rangle` with probability :math:`|\alpha|^2` and the outcome :math:`|1\rangle` with probability :math:`|\beta|^2`. In order to get accurate statistics on the measurement outcomes, the quantum circuit needs to be executed multiple times (shots) to gather sufficient data. The measurement process can be mathematically represented using projection operators. :cite:p:`NC10` For a single qubit, the projection operators for measuring in the computational basis (Pauli-Z) are:

.. math::
    P_0 = |0\rangle\langle0| = \begin{pmatrix}
    1 & 0 \\
    0 & 0
    \end{pmatrix}, \quad P_1 = |1\rangle\langle1| = \begin{pmatrix}
    0 & 0 \\
    0 & 1
    \end{pmatrix}

As the purpose of this package is to try and emulate real hardware, the only outputs from the simulation of quantum circuits are probabilistic measurement results and it is currently not possible to extract the quantum state after operations without measurement. Especially when simulating the circuit under noise, it is not possible to extract a quantum state due to mixture of states as explained in :doc:`quantum_noise`.