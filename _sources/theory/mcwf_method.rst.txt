Monte-Carlo Wave Function (MCWF) Method
=========================================
The quantum jump, or Monte-Carlo Wave Function (MCWF), method first introduced in 1993 to simulate open quantum systems in quantum optics but can be extended (or modified) to solve any open quantum system. It is a technique used to simulate open quantum systems by unraveling the Lindblad master equation shown in :doc:`quantum_noise` into a statistical ensemble of quantum trajectories. The modified Lindblad master equation is given by: :cite:p:`MCD93`

.. math::
    \dot{\rho} = \frac{1}{i\hbar}[H, \rho] + \sum_m (J_m\rho J_m^\dagger - \frac{1}{2}\{J_mJ_m^\dagger, \rho\})

where :math:`J_m` are called the "Jump" operators. The main idea of the MCWF method is that instead of evolving the density matrix (as performed using the Kraus representation), we simulate many stochastic trajectories starting from the initial state of the wavefunction :math:`|\psi(t)\rangle` undergoing a continuous non-Hermitian evolution under an effective Hamiltonian :math:`H_{eff}` punctuated by quantum jumps where one of the operators :math:`J_m` act on the wavefunction. The expectation value of any observable :math:`A` is computed by the ensemble average of many such trajectories (with :math:`N` being the number of trajectories): :cite:p:`MCD93`

.. math::
    H_{eff} = H - \frac{-i\hbar}{2} \sum_m J_m J_m^\dagger

.. math::
    \langle A \rangle (t) = \frac{1}{N} \sum_{n=1}^N \langle \psi_n(t) | A | \psi_n(t) \rangle

In context of quantum circuits, the MCWF method can be used to simulate the effect of noise channels on quantum states by defining appropriate jump operators for each noise channel and as the number of trajectories sampled increases, the results converge to the fully evolved density matrix of the system.

.. math::
    lim_{N\to\infty} \frac{1}{N} \sum_{n=1}^N |\psi_n(t)\rangle\langle\psi_n(t)| = \rho(t)

For more information on the MCWF method, refer to the original paper in :cite:t:`MCD93`.

As the probabilities of the different states are obtained from the diagonal elements of the density matrix, the MCWF method can just computed the state probabilities by averaging over the measurement results of many trajectories, making it computationally more efficient than evolving the full density matrix, or even computing the density matrix for each trajectory and storing it for averaging later (memory efficiency).