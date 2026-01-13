"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using pennylane as a quantum circuit simulator backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel (shared/distributed memory) environment.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pennylane as qml
from pennylane import numpy as np
import numpy as npy
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
    probs = npy.zeros(len(sparse_matrix_list), dtype=np.float64)
    for k, sparse_matrix in enumerate(sparse_matrix_list):
        data, indices, indptr = sparse_matrix
        res = npy.zeros_like(state)
        for i in range(res.shape[0]):
            for j in range(indptr[i], indptr[i+1]):
                res[i] += data[j] * state[indices[j]]
        probs[k] = npy.vdot(res, res).real
    return probs

@njit(fastmath=False)
def update_statevector(sparse_matrix:tuple[np.ndarray, int, int],
                       state:np.ndarray[np.complex128],
                       prob:float)->np.ndarray[np.complex128]:
    """
    Perfroms the matrix-vector product for sparse matrices in CSR format and provides the updated statevector under a given noise trajectory.

    Args:
        sparse_matrix (tuple[np.ndarray]): Sparse matrix in CSR format representing the noise operator.
        state (np.ndarray[np.complex128]): Current statevector of the quantum system.
        prob (float): Probability of the state evolving under the given noise operator.
    
    Returns:
        np.ndarray[np.complex128]: Updated statevector after applying the noise operator (after normalization).
    """
    data, indices, indptr = sparse_matrix
    res = npy.zeros_like(state)
    for i in range(res.shape[0]):
        for j in range(indptr[i], indptr[i+1]):
            res[i] += data[j] * state[indices[j]]
    return res / npy.sqrt(prob)


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
        Constructor for the Remote Executor class.

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
        
        self.dev = qml.device("lightning.qubit", wires=self.num_qubits)
        
        self.param_gates = {"rz", "rx", "unitary", "rzz"}
        self.instruction_map = {
            "x" : qml.X,
            "sx" : qml.SX,
            "rz" : qml.RZ,
            "rx" : qml.RX,
            "ecr" : qml.ECR,
            "cz" : qml.CZ,
            "rzz" : qml.IsingZZ,
            "unitary": qml.QubitUnitary,
        }
        
        self._setup_qnodes()
        self._create_gate_handlers()

    def _setup_qnodes(self):
        """Pre-compile qnodes for better performance"""
        @qml.qnode(device=self.dev)
        def apply_gate_noparams(state:np.ndarray[np.complex128], 
                                gate_op:callable, 
                                qubit_list:list[int])->np.ndarray[np.complex128]:
            """
            Apply Non-parameteric gates to the quantum circuit.

            Args:
                state (np.ndarray[np.complex128]): Current state of the qubit system.
                gate_op (callable): Operator to apply on the qubit system.
                qubit_list (list[int]): List of qubits that the operator must bbe applied to.
            
            Returns:
                np.ndarray[np.complex128]: The updated state of the qubit system.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            gate_op(qubit_list)
            return qml.state()
            
        @qml.qnode(device=self.dev)
        def apply_gate_params(state:np.ndarray[np.complex128], 
                              gate_op:callable, 
                              params:np.ndarray[np.complex128], 
                              qubit_list:list[int])->np.ndarray[np.complex128]:
            """
            Apply Parameteric gates to the quantum circuit

            Args:
                state (np.ndarray[np.complex128]): Current state of the qubit system.
                gate_op (callable): Operator to apply on the qubit system.
                params (np.ndarray[np.complex128]): The value for the parameterized gate.
                qubit_list (list[int]): List of qubits that the operator must bbe applied to.

            Returns:
                np.ndarray[np.complex128]: The updated state of the qubit system.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            gate_op(params, qubit_list)
            return qml.state()
            
        @qml.qnode(device=self.dev)
        def get_probs(state:np.ndarray[np.complex128])->np.ndarray[np.float64]:
            """
            Compute the final probabilties of the qubit system in the trajectory.

            Args:
                state (np.ndarray[np.complex128]): Final state of the qubit system in the current trajectory.

            Returns:
                np.ndarray[np.float64]: Probabilities of the measured qubits from the qubit system.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            return qml.probs(wires=self.measured_qubits)
            
        self.apply_gate_noparams = apply_gate_noparams
        self.apply_gate_params = apply_gate_params
        self.get_probs = get_probs

    def _create_gate_handlers(self):
        """Create gate handler functions to eliminate conditionals in main loop"""
        
        def safe_apply_gate_noparams(state:np.ndarray[np.complex128], 
                                     gate_op:callable, 
                                     qubits:list[int])->np.ndarray[np.complex128]:
            """
            Apply gate with NaN handling only when needed.

            Args:
                state (np.ndarray[np.complex128]): Current state of the qubit system.
                gate_op (callable): Operator to apply on the qubit system.
                qubits (list[int]): List of qubits the the operator is applied to.

            Returns:
                np.ndarray[np.complex128]: Updated state afer NaN checks.
            """
            psi_dash = self.apply_gate_noparams(state, gate_op, qubits)
            if np.isnan(psi_dash).any():
                noisy_state = state + np.random.normal(0, 1e-8, size=state.shape)
                psi_dash = self.apply_gate_noparams(noisy_state, gate_op, qubits)
            return psi_dash
        
        def handle_two_qubit_gates(state:np.ndarray[np.complex128], 
                                   gate:str, 
                                   qubits:list[int], 
                                   params:np.ndarray[np.complex128])->np.ndarray[np.complex128]:
            """
            Apply two qubit gate with NaN handling when needed.

            Args:
                state (np.ndarray[np.complex128]): Current state of the qubit system.
                gate (str): Name of the gate to be applied to the system.
                qubits (list[int]): List of qubits that the gate must be applied to.
                params (np.ndarray[np.compplex128]): Parameter value for parameterized two qubit gates.

            Returns:
                np.ndarray[np.complex128]: Updated state after NaN checks.
            """
            qpair = tuple(qubits)
            psi_dash = safe_apply_gate_noparams(state, self.instruction_map[gate], qubits)
            ops = self.two_qubit_noise[self.two_qubit_noise_index[gate]][1][qpair]["operators"]
            kraus_probs = compute_trajectory_probs(ops, psi_dash)
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            del kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            return update_statevector(ops[kraus_idx], psi_dash, kraus_probs[kraus_idx])
            
        def handle_param_gate(state:np.ndarray[np.complex128], 
                              gate:str, 
                              qubits:list[int], 
                              params:np.ndarray[np.complex128])->np.ndarray[np.complex128]:
            """
            Apply parameterized gates with NaN handling.

            Args:
                state (np.ndarray[np.complexy128]): Current state of the qubit system.
                gate (str): Name of the gate to be applied to the qubit system.
                qubits (list[int]): List of qubits to applied the gate to.
                params (np.ndarray[np.complex128]): Parameter values to apply.
            
            Returns:
                np.ndarray[np.complex128]: Updated state of the system after NaN checks.
            """
            result = self.apply_gate_params(state, self.instruction_map[gate], params, qubits)
            if np.isnan(result).any():
                noisy_state = state + np.random.normal(0, 1e-8, size=state.shape)
                result = self.apply_gate_params(noisy_state, self.instruction_map[gate], params, qubits)
            return result
            
        def handle_single_qubit_noise(state:np.ndarray[np.complex128], 
                                      gate:str, 
                                      qubits:list[int], 
                                      params:np.ndarray[np.complex128])->np.ndarray[np.complex128]:
            """
            Applies the noise operator to the system for single qubit gates.

            Args:
                state (np.ndarray[np.complex128]): Current State of the qubit system.
                gate (str): Name of the gate to apply.
                qubits (list[int]): Qubit to which the gate is applied.
                params (np.ndarray[np.complex128]): Value for the parameterized gate.

            Returns:
                np.ndarray[np.complex128]: Updated state of the system after noise application.
            """
            psi_dash = safe_apply_gate_noparams(state, self.instruction_map[gate], qubits)
            ops = self.single_qubit_noise[qubits[0]][1][gate]["kraus_operators"]
            kraus_probs = compute_trajectory_probs(ops, psi_dash)
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            del kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            return update_statevector(ops[kraus_idx], psi_dash, kraus_probs[kraus_idx])
        
        self.gate_handlers = {}
        for gate in self.instruction_map:
            if gate in ["ecr", "cz"]:
                self.gate_handlers[gate] = handle_two_qubit_gates
            elif gate in self.param_gates:
                self.gate_handlers[gate] = handle_param_gate
            else:
                self.gate_handlers[gate] = handle_single_qubit_noise

    def run(self, 
            trajectories:int,
            instruction_list:list,
            measured_qubits:list[int])->np.ndarray[np.float64]:
        """
        Main method of the module to execute the MCWF trajectory.

        Args:
            trajectories (int): Total number of trajectories that need to be run by the core.
            instruction_list (list): List of instructions to build the quantum circuit.
            measured_qubits (list[int]): List of qubits to measure.
        
        Returns:
            np.ndarray[np.float64]: Probabilities of the measured qubits after the circuit execution.
        """
        self.instruction_list = instruction_list
        self.measured_qubits = measured_qubits
        
        def compute_trajectory(traj_id:int)->np.ndarray[np.float64]:
            """
            Computes the probabilities of the current MCWF trajectory.

            Args:
                traj_id (int): Trajectory ID.
            
            Returns:
                np.ndarray[np.float64]: Probabilities from the current trajectory.
            """
            np.random.seed(42 + traj_id)
            init_state = np.zeros(2**self.num_qubits, dtype=np.complex128)
            init_state[0] = 1.0
            state = init_state.copy()
            
            for gate, qubits, params in self.instruction_list:
                gc.collect()
                state = self.gate_handlers[gate](state, gate, qubits, params)
                if np.isnan(state).any():
                    return np.zeros(2**len(self.measured_qubits))
                gc.collect()
            
            probs = self.get_probs(state)
            del state
            gc.collect()
            return np.zeros(probs.shape) if np.isnan(probs).any() else probs
        
        result = np.zeros(2**len(self.measured_qubits), dtype=np.float64)
        for traj_id in range(trajectories):
            result += compute_trajectory(traj_id)
        return result