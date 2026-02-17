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
    return probs

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

@ray.remote
class RemoteExecutor:
    """
    Module that performs the parallel execution of the quantum circuit according to the Monte-Carlo Wavefunction method.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 two_qubit_noise_index:dict
                 )->None:
        """
        Constructor for the RemoteExecutor class.

        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            single_qubit_noise (dict): Dictionary containing the noise operators for single qubit gates.
            two_qubit_noise (dict): Dictionary containing the noise operators for two qubit gates.
            two_qubit_noise_index (dict): Dictionary mapping qubit pairs to their indices in the two qubit noise dictionary.
            instruction_list (list): List of instructions to build the quantum circuit.
            measured_qubits (list[int]): List of qubits to measure.
        """
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = two_qubit_noise
        self.two_qubit_noise_index = two_qubit_noise_index
        self.probs_sum = np.zeros(2**self.num_qubits, dtype=np.float64)
        exp = lambda x: np.exp(1j * x)
        self.instruction_map = {
            "x": lambda q, p: gate.X(q[0]),
            "sx": lambda q, p: gate.sqrtX(q[0]),
            "rz": lambda q, p: gate.RotZ(q[0], p),
            "rx": lambda q, p: gate.RotX(q[0], p),
            "cz": lambda q, p: gate.CZ(q[0], q[1]),
            "ecr": lambda q, p: gate.DenseMatrix(q, (1 / np.sqrt(2)) * np.array([[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]])),
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
            "rzz": self._no_noise,
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
        kraus_probs = compute_trajectory_probs(ops, state.copy())
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        new_state =  update_statevector(ops[chosen_idx], state, kraus_probs[chosen_idx])
        return new_state / np.linalg.norm(new_state)
    
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
        ops = self.two_qubit_noise[self.two_qubit_noise_index[gate_name]][1][qubit_pair]["operators"]
        kraus_probs = compute_trajectory_probs(ops, state)
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        new_state = update_statevector(ops[chosen_idx], state, kraus_probs[chosen_idx])
        return new_state / np.linalg.norm(new_state)
    
    def _no_noise(self,
                  state:np.ndarray[np.complex128],
                  gate_name:str,
                  qubit_index:list[int])->np.ndarray[np.complex128]:
        """
        Private method that returns the statevector unchanged when no noise is to be applied.
        """
        return state

    def run(self,
            traj_id:int,
            instruction_list:list[list[str, list[int], float|None]]
        )->None:
        """
        Main method of the module to execute the MCWF trajectories.

        Args:
            traj_id (int): Trajectory ID for the simulation.
            instruction_list (list[list[str, list[int], float|None]]): List of instructions to build the quantum circuit.
        """
        self.instruction_list = instruction_list

        def compute_trajectory(traj_id:int)->np.ndarray[np.float64]:
            """
            Method to compute a single MCWF trajectory.

            Args:
                traj_id (int): Trajectory ID for the simulation.

            Returns:
                np.ndarray[np.float64]: Probabilities after executing the trajectory.
            """
            np.random.seed(42 + traj_id)
            init_state = np.zeros(2**self.num_qubits, dtype=np.complex128)
            init_state[0] = 1.0 + 0.0j
            for gate_name, qubit_index, parameters in self.instruction_list:
                state = QuantumState(self.num_qubits)
                state.load(init_state)
                circuit = QuantumCircuit(self.num_qubits)
                circuit.add_gate(self.instruction_map[gate_name](qubit_index, parameters))
                circuit.update_quantum_state(state)
                state_vector = state.get_vector()
                state_vector_updated = self.noise_function_map[gate_name](state_vector.copy(), gate_name, qubit_index)
                init_state = state_vector_updated
            return init_state

        final_state = compute_trajectory(traj_id)
        self.probs_sum += np.abs(final_state)**2

    def get(self,
            measured_qubits:list[int])->np.ndarray[np.float64]:
        """
        Method to get the accumulated probabilities after all trajectories have been run.

        Args:
            measured_qubits (list[int]): List of qubits that were measured.

        Returns:
            np.ndarray[np.float64]: Accumulated probabilities after all trajectories.
        """
        return self.probs_sum
    
    def reset(self, 
              measured_qubits:list[int])->None:
        """
        Method to reset the accumulated probabilities and the measured qubits.

        Args:
            measured_qubits (list[int]): List of qubits that will be measured.
        """
        self.probs_sum = np.zeros(2**self.num_qubits, dtype=np.float64)
        