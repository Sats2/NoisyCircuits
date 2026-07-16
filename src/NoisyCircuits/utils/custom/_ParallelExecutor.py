# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using the custom C++ simulation backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel environment. The parallel environment is available for both shared memory as well as distributed memory systems. 

The shared memory parallel execution runs a single trajectory per core and does not a single trajectory across multiple cores. The class RemoteExecutor is used to run the simulations in parallel on a shared memory system. Unlike the case with pennylane/qiskit/qulacs solvers, this does not use Ray but instead relies on OpenMP for parallelization and is directly integrated into the C++ simulation backend. The class name is unchanged from the other backend in order to maintain a consistent interface for the QuantumCircuit module.
"""
import numpy as np
import simulator


class RemoteExecutor:
    """
    Wrapper to the module that performs the parallel execution of the quantum circuit according to the Monte-Carlo Wavefunction method.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 num_cores:int=2)->None:
        """
        Constructor for the RemoteExecutor class.

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.
        single_qubit_noise : dict
            Dictionary containing the single qubit noise parameters.
        two_qubit_noise : dict
            Dictionary containing the two qubit noise parameters.
        num_cores : int
            Number of cores to use for parallel execution. Defaults to 2.
        """
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = two_qubit_noise
        self.num_cores = num_cores

    def run(self,
            num_trajectories:int,
            instruction_list:list[list[str, list[int], float]]
            )->np.ndarray[np.float64]:
        """
        Method that calls the C++ simulation backend.

        Parameters
        ----------
        num_trajectories : int
            Number of trajectories to run in parallel.
        instruction_list : list[list[str, list[int], float]])
            List of instructions to build the quantum circuit.
        qubits : list[int]
            List of qubits that should be measured.

        Returns
        -------
        np.ndarray[np.float64]
            Output probabilities of the noisy quantum circuit simulation.
        """
        output_array = np.zeros(1 << self.num_qubits, dtype=np.complex128)
        simulator.simulate_circuit(instruction_list, output_array, self.single_qubit_noise, self.two_qubit_noise, self.num_qubits, True, num_trajectories, False, self.num_cores)
        return output_array.astype(np.float64)