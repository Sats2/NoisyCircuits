import pennylane as qml
from pennylane import numpy as np


class DensityMatrixSolver:
    """
    Class to solve quantum circuits using density matrices. Assumes that the circuit is defined with the qubit map already implemented.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 ecr_noise:dict,
                 measurement_noise:dict,
                 instruction_list:list,
                 qubit_instruction_list:list)->None:
        """
        Initializes the DensityMatrixSolver with the given parameters.

        Args:
            num_qubits (int): Number of qubits in the circuit.
            single_qubit_noise (dict): Noise instructions for single qubit gates for all qubits used.
            ecr_noise (dict): Noise instructions for entangling gates (ECR) for all qubits used.
            measurement_noise (dict): Noise instructions for measurement operations for all qubits used.
            instruction_list (list): List of instructions to be executed on the circuit.
            qubit_instruction_list (list): List of instructions specific to each qubit.

        Raises:
            TypeError: If any of the input types are incorrect.
            ValueError: If num_qubits is less than 1.
            TypeError: If single_qubit_noise is not a dictionary.
            TypeError: If ecr_noise is not a dictionary.
            TypeError: If measurement_noise is not a dictionary.
            TypeError: If instruction_list is not a list.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if not isinstance(single_qubit_noise, dict):
            raise TypeError("single_qubit_noise must be a dictionary")
        if not isinstance(ecr_noise, dict):
            raise TypeError("ecr_noise must be a dictionary")
        if not isinstance(measurement_noise, dict):
            raise TypeError("measurement_noise must be a dictionary")
        if not isinstance(instruction_list, list):
            raise TypeError("instruction_list must be a list")
        if not isinstance(qubit_instruction_list, list):
            raise TypeError("qubit_instruction_list must be a list")
        if num_qubits < 1:
            raise ValueError("num_qubits must be greater than or equal to 1")
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.ecr_noise = ecr_noise
        self.measurement_noise = measurement_noise
        self.instruction_list = instruction_list
        self.qubit_instruction_list = qubit_instruction_list

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
<<<<<<< HEAD
            instruction_map = {
                "x": lambda q: qml.X(q),
                "sx": lambda q: qml.SX(q),
                "rz": lambda t, q: qml.RZ(t, q),
                "ecr": lambda q: qml.ECR(q),
                "unitary": lambda p, q: qml.QubitUnitary(p, q),
            }
=======
            # Define gate execution functions
            gate_executors = {
                "x": lambda params, qubits: qml.X(qubits),
                "sx": lambda params, qubits: qml.SX(qubits),
                "rz": lambda params, qubits: qml.RZ(params, qubits),
                "ecr": lambda params, qubits: qml.ECR(qubits),
                "unitary": lambda params, qubits: qml.QubitUnitary(params, qubits),
            }
            
            # Define noise application functions
            def apply_ecr_noise(qubits):
                qml.QubitChannel(self.ecr_noise[tuple(qubits)]["qubit_channel"], wires=qubits)
            
            def apply_single_qubit_noise(gate, qubits):
                qml.QubitChannel(self.single_qubit_noise[qubits[0]][gate]["qubit_channel"], wires=qubits)
            
            def no_noise(qubits):
                pass  # No noise applied
            
            # Direct lookup mapping for noise handlers - no conditionals needed
            noise_handlers = {
                "x": lambda qubits: apply_single_qubit_noise("x", qubits),
                "sx": lambda qubits: apply_single_qubit_noise("sx", qubits),
                "rz": lambda qubits: apply_single_qubit_noise("rz", qubits),
                "ecr": apply_ecr_noise,
                "unitary": no_noise,
            }
            
>>>>>>> parent of b9c9302 (Merge pull request #3 from Sats2/heron_chip_extension)
            for entry in self.instruction_list:
                gate_instruction = entry[0]
                qubit_added = entry[1]
                params = entry[2]
                if gate_instruction in ["rz", "unitary"]:
                    instruction_map[gate_instruction](params, qubit_added)
                elif gate_instruction in ["ecr"]:
                    instruction_map[gate_instruction](qubit_added)
                    qml.QubitChannel(self.ecr_noise[tuple(qubit_added)]["operators"], wires=range(self.num_qubits))
                else:
                    instruction_map[gate_instruction](qubit_added)
                    qml.QubitChannel(self.single_qubit_noise[qubit_added[0]][gate_instruction]["kraus_operators"], wires=range(self.num_qubits))
            return qml.probs(wires=qubits)
        
        probs = run_circuit()
        return probs