"""
This module provides the ability for the QuantumCircuit module to perform a pure statevector simulation for a specific quantum circuit using pennylane as a quantum circuit simulation backend. Alternatively, the user can opt to use just this method to perform a pure statevector simulation using a custom instruction set as long as the gates applied belong to the set of gates pre-defined by the QPU basis gates from the IBM Eagle/Heron QPU architectures. It is recommended to run all simulations via the QuantumCircuit module in order to allow for the correct decomposition of quantum gates according to the QPU's basis gates.

Example:
    >>> import numpy as np
    >>> from NoisyCircuits.utils.PureStateSolver import PureStateSolver
    >>> instruction_list = []
    >>> instruction_list.append(["rx", [0], np.pi])
    >>> instruction_list.append(["ecr", [0, 1], None])
    >>> solver = PureStateSolver(num_qubits=2, instruction_list=instruction_list, num_cores=1)
    >>> solver.solve(qubits=[0,1])
    [0.5, 0.5, 0.0, 0.0]

This module contains only one class `PureStateSolver` which has only one callable function `solve(qubits)` whose arguements are the qubits that are to be measured.
"""
import pennylane as qml
from pennylane import numpy as np


class PureStateSolver:
    """
    Class to solve quantum circuits using pure statevector simulations without noise.
    """
    def __init__(self,
                 num_qubits:int,
                 instruction_list:list,
                 num_cores:int,
                 return_statevector:bool
                )->None:
        """
        Initializes the PureStateSolver.

        Parameters:
        -----------
        num_qubits : int
            The number of qubits in the circuit.
        instruction_list : list
            The list of instructions to be applied.
        num_cores : int, optional
            The number of CPU cores to use for parallel processing (default is 1).
        
        Notes:
        ------
        The input argument `num_cores` is unused with pennylane as a solver backend as pennylane does not support parallel statevector simulation.
        """
        self.num_qubits = num_qubits
        self.instruction_list = instruction_list
        self.return_statevector = return_statevector
        
    def solve(self)->np.ndarray[np.float64]|np.ndarray[np.complex128]:
        """
        Performs the quantum circuit simulation using pure statevector methods.

        Returns
        -------
        np.ndarray
            The probabilities of measuring the specified qubits in the computational basis.
        """
        dev = qml.device("lightning.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def run_circuit(qubits:list[int], 
                        return_statevector:bool=self.return_statevector
                        )->np.ndarray[np.float64] | np.ndarray[np.complex128]:
            """
            Function to run the quantum circuit defined by the instruction list.

            Parameters
            ----------
            qubits : list[int]
                The list of qubits for which to compute the probabilities.
            return_statevector : bool
                Whether to return the statevector or the probabilities (default is False).

            Returns
            -------
            np.ndarray[np.float64] | np.ndarray[np.complex128]
                The probabilities of measuring the specified qubits in the computational basis if return_statevector is False, otherwise the statevector of the quantum circuit.
            """
            instruction_map = {
                "x": lambda q: qml.X(q[0]),
                "sx": lambda q: qml.SX(q[0]),
                "rz": lambda t, q: qml.RZ(t, q[0]),
                "rx": lambda t,q: qml.RX(t, q[0]),
                "ecr": lambda q: qml.ECR(q),
                "cz": lambda q: qml.CZ(q),
                "rzz": lambda t,q: qml.IsingZZ(t, q),
                "unitary": lambda p, q: qml.QubitUnitary(p, q),
            }
            for entry in self.instruction_list:
                gate_instruction = entry[0]
                qubit_added = entry[1]
                params = entry[2]
                if gate_instruction in ["rz", "rx", "unitary", "rzz"]:
                    instruction_map[gate_instruction](params, qubit_added)
                else:
                    instruction_map[gate_instruction](qubit_added)
            if return_statevector:
                return qml.state()
            else:
                return qml.probs(wires=qubits)
        
        output = run_circuit(self.num_qubits)
        return output