"""
This module provides the ability for the QuantumCircuit module to perform a density matrix simulation for a specific quantum circuit using pennylane as a quantum circuit simulation backend. Alternatively, the user can opt to use just this method to perform a density matrix simulation using a custom instruction set with a custom noise model for single qubit and two qubit gates, as long as the gates applied belong to the set of gates pre-defined by the QPU basis gates from the IBM Eagle/Heron QPU architectures. The measurement error is not considered in this module and is only applied from within the QuantumCircuit module. For the full application of noise information from the quantum hardware, it is recommended to run all simulations via the QuantumCircuit module. 

Example:
    >>> import numpy as np
    >>> from NoisyCircuits.utils.DensityMatrixSolver import DensityMatrixSolver
    >>> instruction_list = []
    >>> instruction_list.append(["rx", [0], np.pi])
    >>> instruction_list.append(["ecr", [0, 1], None])
    >>> solver = DensityMatrixSolver(num_qubits=2, single_qubit_noise=single_qubit_noise, two_qubit_noise=two_qubit_noise, instruction_list=instruction_list)
    >>> solver.solve(qubits=[0,1])
    [0.45, 0.45, 0.05, 0.05]

This module contains only one class `DensityMatrixSolver` which has only one callable function `solve(qubits)` whose arguements are the qubits that are to be measured.
"""

import pennylane as qml
from pennylane import numpy as np


class DensityMatrixSolver:
    """
    Class to solve quantum circuits using density matrices. Assumes that the circuit is defined with the qubit map already implemented.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 instruction_list:list)->None:
        """
        Initializes the DensityMatrixSolver with the given parameters.

        Args:
            num_qubits (int): Number of qubits in the circuit.
            single_qubit_noise (dict): Noise instructions for single qubit gates for all qubits used.
            two_qubit_noise (dict): Noise instructions for entangling gates for all qubits used.
            instruction_list (list): List of instructions to be executed on the circuit.

        Raises:
            TypeError: If any of the input types are incorrect.
            ValueError: If num_qubits is less than 1.
            TypeError: If single_qubit_noise is not a dictionary.
            TypeError: If two_qubit_noise is not a dictionary.
            TypeError: If instruction_list is not a list.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if not isinstance(single_qubit_noise, dict):
            raise TypeError("single_qubit_noise must be a dictionary")
        if not isinstance(two_qubit_noise, dict):
            raise TypeError("two_qubit_noise must be a dictionary")
        if not isinstance(instruction_list, list):
            raise TypeError("instruction_list must be a list")
        if num_qubits < 1:
            raise ValueError("num_qubits must be greater than or equal to 1")
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = two_qubit_noise
        self.instruction_list = instruction_list

    def solve(self,
              qubits:list)->np.ndarray:
        """
        Solves the quantum circuit using density matrices and returns the probabilities of measuring each qubit in the computational basis.

        Args:
            qubits (list): List of qubits to be measured.

        Returns:
            np.ndarray: Probabilities of measuring each qubit in the computational basis.
        """
        dev_mixed = qml.device("default.mixed", wires=self.num_qubits)

        @qml.qnode(dev_mixed)
        def run_circuit():
            # Define gate execution functions
            gate_executors = {
                "x": lambda params, qubits: qml.X(qubits),
                "sx": lambda params, qubits: qml.SX(qubits),
                "rz": lambda params, qubits: qml.RZ(params, qubits),
                "rx": lambda params, qubits: qml.RX(params, qubits),
                "ecr": lambda params, qubits: qml.ECR(qubits),
                "cz": lambda params, qubits: qml.CZ(qubits),
                "rzz": lambda params, qubits: qml.IsingZZ(params, qubits),
                "unitary": lambda params, qubits: qml.QubitUnitary(params, qubits),
            }
            
            # Define noise application functions
            def apply_two_qubit_noise(gate, qubits):
                qml.QubitChannel(self.two_qubit_noise[gate][tuple(qubits)]["qubit_channel"], wires=qubits)
            
            def apply_single_qubit_noise(gate, qubits):
                qml.QubitChannel(self.single_qubit_noise[qubits[0]][gate]["qubit_channel"], wires=qubits)
            
            def no_noise(qubits):
                pass  # No noise applied
            
            # Direct lookup mapping for noise handlers - no conditionals needed
            noise_handlers = {
                "x": lambda qubits: apply_single_qubit_noise("x", qubits),
                "sx": lambda qubits: apply_single_qubit_noise("sx", qubits),
                "rz": lambda qubits: apply_single_qubit_noise("rz", qubits),
                "rx": lambda qubits: apply_single_qubit_noise("rx", qubits),
                "ecr": lambda qubits:apply_two_qubit_noise("ecr", qubits),
                "cz": lambda qubits: apply_two_qubit_noise("cz", qubits),
                "rzz": lambda qubits: apply_two_qubit_noise("rzz", qubits),
                "unitary": no_noise,
            }
            
            for entry in self.instruction_list:
                gate_instruction = entry[0]
                qubit_added = entry[1]
                params = entry[2]
                
                # Execute gate and apply noise using direct lookup - no conditionals
                gate_executors[gate_instruction](params, qubit_added)
                noise_handlers[gate_instruction](qubit_added)
            
            return qml.probs(wires=qubits)
        
        probs = run_circuit()
        return probs