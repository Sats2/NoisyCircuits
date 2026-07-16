# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module provides the ability for the QuantumCircuit module to perform a pure statevector simulation for a specific quantum circuit using the custom-built C++ simulation as a qauntum circuit simulation backend. Alternatively, the user can opt to use just this method to perform a pure statevector simulation using a custom instruction set as long as the gates applied belong to the set of gates pre-defined by the QPU basis gates from the IBM Eagle/Heron QPU architectures. It is recommended to run all simulation via the QuantumCircuit module in order to allow for the correct decomposition of quantum gates according to the QPU's basis gates.

Example:
    >>> import numpy as np
    >>> from NoisyCircuits.utils.solvers import load_solver
    >>> instruction_list = []
    >>> instruction_list.append(["rx", [0], np.pi])
    >>> instruction_list.append(["cz", [0, 1], None])
    >>> solver = load_solver("custom")
    >>> pure_state_solver = solver.PureStateSolver(num_qubits=2, instruction_list=instruction_list, num_cores=2, return_statevector=False)
    >>> pure_state_solver.solve(qubits=[0, 1])
    [0.0, 0.0, 1.0, 0.0]
    >>> pure_state_solver.return_statevector = True
    >>> pure_state_solver.solve(qubits=[0, 1])
    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j]

This module contains only one class `PureStateSolver` which has only one callable function `solve(qubits)` whose arguements are the qubits that are to be measured.
"""
import numpy as np
import simulator


class PureStateSolver:
    """
    Class to solve quantum circuits using pure statevector simulations without noise.
    """
    def __init__(self,
                num_qubits:int,
                instruction_list:list,
                num_cores:int=1,
                return_statevector:bool=False)->None:
        """
        Initializes the PureStateSolver.

        Parameters
        ----------
        num_qubits : int
            The number of qubits in the circuit.
        instruction_list : list
            The list of instructions to be applied.
        num_cores : int
            The number of CPU cores to use for the simulation. Defaults to 1.
        return_statevector : bool
            A flag indicating whether to return the full statevector or just the probabilities. Defaults to False.
        """
        self.num_qubits = num_qubits
        self.instruction_list = instruction_list
        self.num_cores = num_cores
        self.return_statevector = return_statevector

    def solve(self)->np.ndarray[np.float64|np.complex128]:
        """
        Performs the quantum circuit simulation using pure statevector methods.
        
        Returns
        -------
        np.ndarray
            Return an array of either the probabilities of the specified qubits in the computational basis or the full statevector depending on the value of `return_statevector`.
        """
        output_vector = np.zeros((1 << self.num_qubits), dtype=np.complex128)
        simulator.simulate_circuit(self.instruction_list, output_vector, {}, {}, self.num_qubits, False, 1, self.return_statevector, self.num_cores)
        if self.return_statevector:
            print("Returning full statevector.")
            return output_vector
        else:
            return output_vector.real