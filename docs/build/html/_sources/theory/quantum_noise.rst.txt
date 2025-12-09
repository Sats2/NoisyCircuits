Quantum Noise
================

In the NISQ (Noisy Intermediate Scale Quantum) era, quantum computers are prone to various types of noise that can significantly affect their performance. Understanding and mitigating quantum noise is crucial for the development of reliable quantum algorithms and applications. This noise cannot be modelled as unitary operations, as they introduce errors that cannot be reversed simply by applying another unitary transformation but non-unitary operations cannot be used to describe closed quantum systems (i.e., evolve a pure state vector). Thus, there is a requirement for a different representation of qubits that allows for non-unitary operations. This representation is called the density matrix formalism. :cite:p:`NC10`

Density Matrix
----------------
Density Matrices are quantum mechanical frameworks for representing wavefunctions of a noisy (or open) quantum system that deal with mixed states or partial information. A so called mixed state is a statistical ensemble of several pure states with some uncertainty that is not in superposition. A density matrix is defined as:

.. math::
    \rho = \sum_i p_i|\psi_i\rangle\langle\psi_i|

where :math:`p_i` is the probability of the system being in the pure state :math:`|\psi_i\rangle`. The density matrix formalism allows for the description of both pure states and mixed states. A pure state can be represented as a density matrix by setting :math:`p_1 = 1` and all other probabilities to zero. :cite:p:`NC10`

The density matrix has some important properties. The density matrix must be Hermitian (:math:`\rho = \rho^\dagger`), have a trace of one (:math:`Tr(\rho) = 1`), and be positive semi-definite (:math:`\langle v|\rho| v\rangle\geq\;\forall v`). These properties ensure that the density matrix represents a valid quantum state. :cite:p:`tund`

For closed systems, the evolution of the density matrix is defined by the Schrödinger equation writen in the density matrix formalism as:

.. math::
    i\hbar\frac{d}{dt}\rho = [\hat{H}, \rho] \\
    \rho' = \varepsilon(\rho) = U\rho U^\dagger

where :math:`\hat{H}` is the Hamiltonian of the system, :math:`U` is the unitary operator representing the evolution of the system, :math:`\rho'` is the density matrix after the evolution, and :math:`\varepsilon(\rho)` is a operator that maps the initial state to the final state. The square brackets denote the commutator operation (:math:`[A, B] = AB - BA`). :cite:p:`tund` Unfortunately, for noisy (or open) quantum systems, the evolution cannot be described solely by unitary operations and the Schrödinger equation is no longer sufficient. The evolution of open systems is instead described by the `Lindblad master equation <https://en.wikipedia.org/wiki/Lindbladian>`_.

.. math::
    \dot{\rho} = \frac{1}{i\hbar}[H, \rho] + \sum_{n,m} \left( A_n\rho A_m^\dagger - \frac{1}{2}\{ A_m^\dagger A_n, \rho \} \right)

where :math:`A_n` are the Lindblad operators that describe the interaction between the system and its environment, and the curly brackets denote the anti-commutator operation (:math:`\{A, B\} = AB + BA`). :cite:p:`master_equation`


Operator-Sum Representation
--------------------------------
An alternative way to represent the evolution of open quantum systems to think of this system as a larger closed system that includes both the quantum system of interest and its environment. The system's interaction with the environment can be modelled using unitary operations on the combined system. This approach is called the Stinespring dilation theorem which essentially states that for an operator :math:`\varepsilon\rightarrow \mathcal{B}(\mathbb{H_s})` (where :math:`\mathcal{B}(\mathbb{H_s})` is the space of bounded linear operators on the Hilbert space of the system :math:`\mathbb{H_s}`) that is a completely positive and trace-preserving map, then there exists an auxillary Hilbert space, with ancilla initialized with :math:`|0\rangle` and a unitary operator :math:`U` acting on the combined system such that the evolution of the combined system can be described by the Schrödinger equation. :cite:p:`NC10`

Utilizing this theorem, a unitary :math:`U\in \mathbb{C}^{n}\otimes\mathbb{C}^{m}` can be defined that acts on the combined system consisting of the qubit system (of size :math:`n`) and the environment (of size :math:`m`). The evolution of this combined system can be described by the Schrödinger equation. The state of the combined system is given by :math:`\Psi\rangle= |\psi_s\rangle\otimes|\psi_e\rangle` where :math:`|\psi_s\rangle` is the state of the qubit system and :math:`|\psi_e\rangle` is the state of the environment. The evolution of the combined system is given by: :cite:p:`NC10`

.. math::
    |\Psi\rangle = U(\psi_s\rangle\otimes|\psi_e\rangle) = \sum_k \sqrt{p_k} K_k |\psi_s\rangle\otimes|e_k\rangle

where :math:`|e_k\rangle` are the orthonormal basis states of the environment, :math:`p_k` are the probabilities associated with each noise interaction, and :math:`E_k` are the operators acting on the qubit system that describe the individual noise events within the combined unitary :math:`U`. The density matrix of the combined system is given by :math:`\rho_{se} = |\Psi\rangle\langle\Psi|`. To obtain the density matrix of the qubit system alone, the environment is traced out using the partial trace operation. The partial trace over the environment yields the reduced density matrix of the qubit system: :cite:p:`NC10`

.. math::
    \begin{align*}
    \rho_s &= Tr_e(\rho_{se}) \\
    &= Tr_e(U(\rho_s\otimes|0\rangle\langle 0|)U^\dagger) \\
    &= \sum_k (\mathcal{I}_s \otimes \langle k|) U (\rho_s \otimes |0\rangle\langle 0|) U^\dagger (\mathcal{I}_s \otimes |k\rangle)
    \end{align*}

A Kraus operator :math:`K_k` acting on the qubit system's Hilbert space can be defined as: :cite:p:`NC10`

.. math::
    K_k = (\mathcal{I}_s \otimes \langle k|) U (\mathcal{I}_s \otimes |0\rangle)

Using these Kraus operators, the evolution of the qubit system's density matrix can be expressed in the operator-sum representation as: :cite:p:`NC10`

.. math::
    \rho_s' = \sum_k K_k \rho_s K_k^\dagger