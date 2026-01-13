"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using qulacs as a quantum circuit simulator backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel (shared/distributed memory) environment.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from qulacs import QuantumState, QuantumCircuit
import qulacs.gate as gate
import numpy as np
import ray
from numba import njit
import gc


@njit(fastmath=False)
def compute_trajectory_probs(sparse_matrix_list:list[tuple[np.ndarray[np.complex128], int, int]],
                             state:np.ndarray[np.complex128])->np.ndarray[np.float64]:
    """
    Computes the probabilities of the given statevector evolving under noise operators.

    Args:
        sparse_matrix_list (list[tuple[np.ndarray[np.complex128], int, int]]): List of sparse matrices representing the noise operators in CSR format.
        state (np.ndarray[np.complex128]): Statevector of the quantum system.
    
    Returns:
        np.ndarray[np.float64]: Probabilities of the statevector picking a given noise operator.
    """
    probs = np.zeros(len(sparse_matrix_list), dtype=np.float64)
    for k, sparse_matrix in enumerate(sparse_matrix_list):
        data, indices, indptr = sparse_matrix
        res = np.zeros_like(state)
        for i in range(res.shape[0]):
            for j in range(indptr[i], indptr[i+1]):
                res[i] += data[j] * state[indices[j]]
        probs[k] = np.vdot(res, res).real
    return probs/np.sum(probs) if np.sum(probs) == 0.0 or np.isnan(probs).any() else np.ones(len(probs), dtype=np.float64)/len(probs)

@njit(fastmath=False)
def update_statevector(sparse_matrix:tuple[np.ndarray[np.complex128], int, int],
                       state:np.ndarray[np.complex128],
                       prob:float)->np.ndarray[np.complex128]:
    """
    Performs the matrix-vector product for sparse matrices in CSR format and provides the updated statevector under a given noise trajectory.

    Args:
        sparse_matrix (tuple[np.ndarray[np.complex128], int, int]): Sparse matrix in CSR format representing the noise operator.
        state (np.ndarray[np.complex128]): Current statevector of the quantum system.
        prob (float): Probability of the state evolving under the given noise operator.
    
    Returns:
        np.ndarray[np.complex128]: Updated statevector after applying the noise operator (after normalization).
    """
    data, indices, indptr = sparse_matrix
    res = np.zeros_like(state)
    for i in range(res.shape[0]):
        for j in range(indptr[i], indptr[i+1]):
            res[i] += data[j] * state[indices[j]]
    return res / np.sqrt(prob)

def get_probabilities(state:np.ndarray[np.complex128],
                      qubits:list[int])->np.ndarray[np.float64]:
    """
    Computes the measurement probabilities for the specified qubits.

    Args:
        state (np.ndarray[np.complex128]): The statevector of the full quantum system.
        qubits (list[int]): The list of qubits for which to compute the probabilities.

    Returns:
        np.ndarray[np.float64]: The probabilities of measuring each qubit in the computational basis.
    """
    state_tensor = state.reshape([2]*int(np.log2(len(state))))
    sum_axes = tuple(i for i in range(state_tensor.ndim) if i not in qubits)
    probs = np.sum(np.abs(state_tensor)**2, axis=sum_axes)
    return probs.flatten()

@ray.remote
class RemoteExecutor:
    """
    Module that performs the parallel execution of the quantum circuit according to the Monte-Carlo Wavefunction method.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 two_qubit_noise_index:dict)->None:
        """
        Constructor for the RemoteExecutor class.

        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            single_qubit_noise (dict): Dictionary containing the noise operators for single qubit gates.
            two_qubit_noise (dict): Dictionary containing the noise operators for two qubit gates.
            two_qubit_noise_index (dict): Dictionary mapping qubit pairs to their indices in the two qubit noise dictionary.
        """
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = two_qubit_noise
        self.two_qubit_noise_index = two_qubit_noise_index
        exp = lambda x: np.exp(1j * x)
        self.instruction_map = {
            "x": lambda q, p: gate.X(q[0]),
            "sx": lambda q, p: gate.sqrtX(q[0]),
            "rz": lambda q, p: gate.RotZ(q[0], p),
            "rx": lambda q, p: gate.RotX(q[0], p),
            "cz": lambda q, p: gate.CZ(q[0], q[1]),
            "ecr": lambda q, p: gate.DenseMatrix(q, (1 / np.sqrt(2)) * np.array([[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]])),
            "rzz": lambda q, p: gate.DenseMatrix(q, np.array([[exp(-p/2), 0, 0, 0], [0, exp(p/2), 0, 0], [0, 0, exp(p/2), 0], [0, 0, 0, exp(-p/2)]])),
            "unitary": lambda q, p: gate.DenseMatrix(q, p) if len(q) > 1 else gate.DenseMatrix(q[0], p)
        }
        self.noise_function_map = {
            "x": self._apply_single_qubit_noise,
            "sx": self._apply_single_qubit_noise,
            "rz": self._no_noise,
            "rx": self._no_noise,
            "cz": self._apply_two_qubit_noise,
            "ecr": self._apply_two_qubit_noise,
            "rzz": self._apply_two_qubit_noise,
            "unitary": self._no_noise
        }
    
    def _apply_single_qubit_noise(self,
                                   state:np.ndarray[np.complex128],
                                   gate_name:str,
                                   qubit_index:list[int]
                                   )->np.ndarray[np.complex128]:
        """
        Private method that applies applies the noise operator for single qubit gates.

        Args:
            state (np.ndarray[np.complex128]): Current statevector of the quantum system.
            gate_name (str): Name of the gate previously applied.
            qubit_index (list[int]): List of qubits that the noise must be applied to.
        
        Returns:
            np.ndarray[np.complex128]: Updated statevector after applying the noise operator.
        """
        ops = self.single_qubit_noise[qubit_index[0]][1][gate_name]["kraus_operators"]
        kraus_probs = compute_trajectory_probs(ops, state)
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        return update_statevector(ops[chosen_idx], state, kraus_probs[chosen_idx])
    
    def _apply_two_qubit_noise(self,
                               state:np.ndarray[np.complex128],
                               gate_name:str,
                               qubit_index:list[int]
                               )->np.ndarray[np.complex128]:
        """
        Private method that applies the noise operator for two qubit gates.

        Args:
            state (np.ndarray[np.complex128]): Current statevector of the quantum system.
            gate_name (str): Name of the gate previously applied.
            qubit_index (list[int]): List of qubits that the noise must be applied to.
        
        Returns:
            np.ndarray[np.complex128]: Updated statevector after applying the noise operator.
        """
        qubit_pair = tuple(qubit_index)
        ops = self.two_qubit_noise[self.two_qubit_noise_index[gate_name][1][qubit_pair]]["operators"]
        kraus_probs = compute_trajectory_probs(ops, state)
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        return update_statevector(ops[chosen_idx], state, kraus_probs[chosen_idx])
    
    def _no_noise(self,
                  state:np.ndarray[np.complex128],
                  gate_name:str,
                  qubit_index:list[int])->np.ndarray[[np.complex128]]:
        """
        Private method that returns the statevector unchanged when no noise is to be applied.
        """
        return state

    def run(self,
            trajectories:int,
            instruction_list:list,
            measured_qubits:list[int])->np.ndarray[np.float64]:
        """
        Main method of the module to execute the MCWF trajectories.

        Args:
            trajectories (int): List of trajectory IDs to be simulated.
            instruction_list (list): List of instructions to build the quantum circuit.
            measured_qubits (list[int]): List of qubits to measure.

        Returns:
            np.ndarray[np.float64]: Sum of probabilities of the measured qubits after the circuit execution after each trajectory.
        """
        self.instruction_list = instruction_list
        self.measured_qubits = measured_qubits

        def compute_trajectory(traj_id:int)->np.ndarray[np.float64]:
            circuit = QuantumCircuit(self.num_qubits)
            state = QuantumState(self.num_qubits)
            state.set_zero_state()
            np.random.seed(42 + traj_id)
            for gate_name, qubit_index, parameters in self.instruction_list:
                gc.collect()
                circuit.add_gate(self.instruction_map[gate_name](qubit_index, parameters))
                circuit.update_quantum_state(state)
                state_vector = state.get_vector()
                state_vector_updated = self.noise_function_map[gate_name](state_vector, gate_name, qubit_index)
                state.load(state_vector_updated)
                del state_vector
                gc.collect()
            final_state = state.get_vector()
            final_probs = get_probabilities(final_state, self.measured_qubits)
            del state, circuit, final_state
            gc.collect()
            return np.zeros(final_probs.shape, dtype=np.float64) if np.isnan(final_probs).any() else final_probs
        
        results = np.zeros(2**len(self.measured_qubits), dtype=np.float64)
        for traj_id in range(trajectories):
            results += compute_trajectory(traj_id)
        return results