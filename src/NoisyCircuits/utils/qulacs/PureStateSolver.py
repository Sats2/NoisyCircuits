"""
This module provides the ability for the QuantumCircuit module to perform a pure statevector simulation for a specific quantum circuit using Qulacs as a quantum circuit simulator backend. Alternatively, the user can opt to use just this method to perform a pure statevector simulation using a custom instruction set as long as the gates applied belong to the set of gates pre-defined by the QPU basis gates from the IBM Eagle/Heron QPU architectures. It is recommended to run all simulations via the QuantumCircuit module in order to allow for the correct decomposition of quantum gates according to the QPU's basis gates.

Example:
    >>> import numpy as np
    >>> from NoisyCircuits.utils.PureStateSolver import PureStateSolver
    >>> instruction_list = []
    >>> instruction_list.append(["rx", [0], np.pi])
    >>> instruction_list.append(["ecr", [0, 1], None])
    >>> solver = PureStateSolver(num_qubits=2, instruction_list=instruction_list)
    >>> solver.solve(qubits=[0,1])
    [0.5, 0.5, 0.0, 0.0]

This module contains only one class `PureStateSolver` which has only one callable function `solve(qubits)` whose arguements are the qubits that are to be measured.
"""
import numpy as np
from qulacs import QuantumCircuit, QuantumState
import qulacs.gate as gate
import gc
from numba import njit

@njit(fastmath=False)
def get_probabilities(state:np.ndarray[np.complex128],
                      qubits:list[int])->np.ndarray[np.float64]:
    """
    Numba JIT-compiled function to compute measurement probabilities for specified qubits.

    Args:
        state(np.ndarray[np.complex128]): The statevector of the full quantum system.
        qubits(list[int]): The list of qubits for which to compute the probabilities.
    
    Returns:
        np.ndarray[np.float64]: The probabilities of measuring each qubit in the computational basis.
    """
    state_tensor = state.reshape([2]*int(np.log2(len(state))))
    sum_axes = tuple(i for i in range(state_tensor.ndim) if i not in qubits)
    probs = np.sum(np.abs(state_tensor)**2, axis=sum_axes)
    return probs.flatten()

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
        circuit = QuantumCircuit(self.num_qubits)
        state = QuantumState(self.num_qubits)
        state.set_zero_state()
        exp = lambda x: np.exp(1j * x)
        instruction_map = {
            "x": lambda q, p: gate.X(q[0]),
            "sx": lambda q, p: gate.sqrtX(q[0]),
            "rz": lambda q, p: gate.RotZ(q[0], p),
            "rx": lambda q, p: gate.RotX(q[0], p),
            "ecr": lambda q, p: gate.DenseMatrix(q, (1 / np.sqrt(2)) * np.array([[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]])),
            "cz": lambda q, p: gate.CZ(q[0], q[1]),
            "rzz": lambda q, p: gate.DenseMatrix(q, np.array([[exp(-p/2), 0, 0, 0], [0, exp(p/2), 0, 0], [0, 0, exp(p/2), 0], [0, 0, 0, exp(-p/2)]])),
            "unitary": lambda q, p: gate.DenseMatrix(q[0], p)
        }
        for entry in self.instruction_list:
            gate_name = entry[0]
            qubit_index = entry[1]
            parameter = entry[2]
            circuit.add_gate(instruction_map[gate_name](qubit_index, parameter))
        circuit.update_quantum_state(state)
        state_array = state.get_vector()
        del state, circuit, instruction_map, exp
        gc.collect()
        return get_probabilities(state_array, qubits)
