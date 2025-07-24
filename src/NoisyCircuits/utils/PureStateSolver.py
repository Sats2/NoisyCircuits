import pennylane as qml
from pennylane import numpy as np


class PureStateSolver:
    """
    Class to solve quantum circuits using pure statevector simulations without noise.
    """
    def __init__(self,
                 num_qubits:int,
                 instruction_list:list)->None:
        """
        Initializes the PureStateSolver.

        Args:
            num_qubits (int): The number of qubits in the circuit.
            instruction_list (list): The list of instructions to be applied.
        """
        self.num_qubits = num_qubits
        self.instruction_list = instruction_list
        
    def solve(self,
              qubits:list[int])->np.ndarray:
        """
        Performs the quantum circuit simulation using pure statevector methods.

        Args:
            qubits (list[int]): The list of qubits for which to compute the probabilities.

        Returns:
            np.ndarray: The probabilities of measuring each qubit in the computational basis.
        """
        dev = qml.device("lightning.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def run_circuit():
            for instruction in self.instruction_list:
                qml.apply(instruction)
            return qml.probs(wires=qubits)
        
        probs = run_circuit()
        return probs