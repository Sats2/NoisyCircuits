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
        def run_circuit(qubits):
            instruction_map = {
                "x": lambda q: qml.X(q),
                "sx": lambda q: qml.SX(q),
                "rz": lambda t, q: qml.RZ(t, q),
                "rx": lambda t,q: qml.RX(t, q),
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
            return qml.probs(wires=qubits)
        
        probs = run_circuit(qubits)
        return probs