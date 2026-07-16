# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module provides the ability for the QuantumCircuit module to perform a pure statevector simulation for a specific quantum circuit using qiskit as a quantum circuit simulation backend. Alternatively, the user can opt to use just this method to perform a pure statevector simulation using a custom instruction set as long as the gates applied belong to the set of gates pre-defined by the QPU basis gates from the IBM Eagle/Heron QPU architectures. It is recommended to run all simulations via the QuantumCircuit module in order to allow for the correct decomposition of quantum gates according to the QPU's basis gates.

Example:
    >>> import numpy as np
    >>> from NoisyCircuits.utils.qiskit import PureStateSolver
    >>> instruction_list = []
    >>> instruction_list.append(["rx", [0], np.pi])
    >>> instruction_list.append(["ecr", [0, 1], None])
    >>> solver = PureStateSolver(num_qubits=2, instruction_list=instruction_list, num_cores=1, return_statevector=False)
    >>> solver.solve(qubits=[0,1])
    [0.5, 0.5, 0.0, 0.0]

This module contains only one class `PureStateSolver` which has only one callable function `solve(qubits)` whose arguements are the qubits that are to be measured.
"""
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector
import numpy as np


class PureStateSolver:
    """
    Class to solve quantum circuits using pure statevector simulations without noise.
    """
    def __init__(self,
                 num_qubits:int,
                 instruction_list:list,
                 num_cores:int,
                 return_statevector:bool)->None:
        """
        Initializes the PureStateSolver.

        Parameters:
        -----------
        num_qubits : int 
            The number of qubits in the circuit.
        instruction_list : list
            The list of instructions to be applied.
        num_cores : int
            The number of CPU cores to use for the simulation.
        return_statevector : bool
            Whether to return the statevector instead of probabilities.
        """
        self.num_qubits = num_qubits
        self.instruction_list = instruction_list
        self.num_cores = num_cores
        self.return_statevector = return_statevector
        
    def solve(self)->np.ndarray[np.float64] | np.ndarray[np.complex128]:
        """
        Performs the quantum circuit simulation using pure statevector methods.

        Returns
        -------
        np.ndarray[np.float64] | np.ndarray[np.complex128]
            The probabilities of measuring the specified qubits in the computational basis. Defaults to None, returning all qubits. If return_statevector is set to True, returns the statevector instead of probabilities.
        """
        circuit = QuantumCircuit(self.num_qubits)
        instruction_map = {
            "x": lambda q, p: circuit.x(q[0]),
            "sx": lambda q, p: circuit.sx(q[0]),
            "rz": lambda q, p: circuit.rz(p, q[0]),
            "rx": lambda q, p: circuit.rx(p, q[0]),
            "ecr": lambda q, p: circuit.ecr(q[0], q[1]),
            "cz": lambda q, p: circuit.cz(q[0], q[1]),
            "rzz": lambda q, p: circuit.rzz(p, q[0], q[1]),
        }
        for entry in self.instruction_list:
            gate_name = entry[0]
            qubit_index = entry[1]
            parameter = entry[2]
            instruction_map[gate_name](qubit_index, parameter)
        circuit.append(SaveStatevector(self.num_qubits), circuit.qubits)
        sim = AerSimulator()
        sim.set_options(max_parallel_threads=self.num_cores)
        res = sim.run(circuit).result()
        state = res.data()["statevector"]
        if self.return_statevector:
            return state
        else:
            return np.abs(state)**2