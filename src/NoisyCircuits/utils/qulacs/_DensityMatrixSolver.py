"""
This module provides the ability for the QuantumCircuit module to perform a density matrix simulation for a specific quantum circuit using qulacs as a quantum circuit simulation backend. Alternatively, the user can opt to use just this method to perform a density matrix simulation using a custom instruction set with a custom noise model for single qubit and two qubit gates, as long as the gates applied belong to the set of gates pre-defined by the QPU basis gates from the IBM Eagle/Heron QPU architectures. The measurement error is not considered in this module and is only applied from within the QuantumCircuit module. For the full application of noise information from the quantum hardware, it is recommended to run all simulations via the QuantumCircuit module. 

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

from qulacs import QuantumCircuit, DensityMatrix
from qulacs import gate
import numpy as np
from qulacs.state import partial_trace

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
              qubits:list[int])->np.ndarray[np.float64]:
        """
        Solves the quantum circuit using density matrix simulation and returns the probabilities of measuring the specified qubits in the computational basis.

        Args:
            qubits (list[int]): List of qubits to be measured.
        
        Returns:
            np.ndarray[np.float64]: Probabilities of measuring each qubit in the computational basis.
        """
        circuit = QuantumCircuit(self.num_qubits)
        state = DensityMatrix(self.num_qubits)
        exp = lambda x: np.exp(1j * x)
        instruction_map = {
            "x": lambda q, p: gate.X(q[0]),
            "sx": lambda q, p: gate.sqrtX(q[0]),
            "rz": lambda q, p: gate.RotZ(q[0], p),
            "rx": lambda q, p: gate.RotX(q[0], p),
            "cz": lambda q, p: gate.CZ(q[0], q[1]),
            "ecr": lambda q, p: gate.DenseMatrix(q, (1 / np.sqrt(2)) * np.array([[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]])),
            "rzz": lambda q, p: gate.DenseMatrix(q, np.array([[exp(-p/2), 0, 0, 0], [0, exp(p/2), 0, 0], [0, 0, exp(p/2), 0], [0, 0, 0, exp(-p/2)]])),
            "unitary": lambda q, p: gate.DenseMatrix(q, p) if len(q) > 1 else gate.DenseMatrix(q[0], p)
        }
        noise_handlers = {
            "x": lambda q: gate.CPTP([gate.DenseMatrix(q[0], 
                                                       self.single_qubit_noise[q[0]]["x"]["qubit_channel"][k]) for k in range(len(self.single_qubit_noise[q[0]]["x"]["qubit_channel"]))]),
            "sx": lambda q: gate.CPTP([gate.DenseMatrix(q[0], 
                                                        self.single_qubit_noise[q[0]]["sx"]["qubit_channel"][k]) for k in range(len(self.single_qubit_noise[q[0]]["sx"]["qubit_channel"]))]),
            "rz": lambda q: gate.CPTP([gate.DenseMatrix(q[0], 
                                                        self.single_qubit_noise[q[0]]["rz"]["qubit_channel"][k]) for k in range(len(self.single_qubit_noise[q[0]]["rz"]["qubit_channel"]))]),
            "rx": lambda q: gate.CPTP([gate.DenseMatrix(q[0], 
                                                        self.single_qubit_noise[q[0]]["rx"]["qubit_channel"][k]) for k in range(len(self.single_qubit_noise[q[0]]["rx"]["qubit_channel"]))]),
            "ecr": lambda q: gate.CPTP([gate.DenseMatrix([q[0], q[1]], 
                                                        self.two_qubit_noise["ecr"][tuple(q)]["qubit_channel"][k]) for k in range(len(self.two_qubit_noise["ecr"][tuple(q)]["qubit_channel"]))]),
            "cz": lambda q: gate.CPTP([gate.DenseMatrix([q[0], q[1]], 
                                                        self.two_qubit_noise["cz"][tuple(q)]["qubit_channel"][k]) for k in range(len(self.two_qubit_noise["cz"][tuple(q)]["qubit_channel"]))]),
            "rzz": lambda q: gate.DenseMatrix(list(q), np.eye(2**len(q))),
            "unitary": lambda q: gate.DenseMatrix(list(q), np.eye(2 ** len(q)))
        }

        for entry in self.instruction_list:
            gate_name = entry[0]
            qubit_index = entry[1]
            parameter = entry[2]
            instruction_map[gate_name](qubit_index, parameter)
            circuit.add_gate(instruction_map[gate_name](qubit_index, parameter))
            circuit.add_gate(noise_handlers[gate_name](qubit_index))
        circuit.update_quantum_state(state)
        if len(qubits) == self.num_qubits:
            probs = np.diag(state.get_matrix()).real
        else:
            trace_qubits = [i for i in range(self.num_qubits) if i not in qubits]
            probs = np.diag(partial_trace(state, trace_qubits).get_matrix()).real
        del state, circuit, instruction_map, noise_handlers, exp
        return probs