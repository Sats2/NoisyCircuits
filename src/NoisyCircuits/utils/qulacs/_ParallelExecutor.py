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
from NoisyCircuits.utils import convert_matrix_to_little_endian
import gc

def get_updated_state_single(gate_op:np.ndarray[np.complex128], 
                             state:np.ndarray[np.complex128], 
                             q:int
                            )->np.ndarray[np.complex128]:
    """
    Function to get the updated state of the qubit system after applying a single qubit noise operator.

    Parameters
    ----------
    gate_op : np.ndarray[np.complex128]
        The noise operator to be applied to the qubit system.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q : int
        The qubit to which the noise operator is applied.

    Returns
    -------
    np.ndarray[np.complex128]
        The updated state of the qubit system after applying the noise operator.
    """
    dim = 1 << int(np.log2(state.size))
    stride = 1 << q
    psi_dash = np.zeros_like(state)
    for p in range(0, dim >> 1, 1):
        i = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1)
        j = i | stride
        psi_dash[i] += gate_op[0,0] * state[i] + gate_op[0,1] * state[j]
        psi_dash[j] += gate_op[1,0] * state[i] + gate_op[1,1] * state[j]
    return psi_dash

def compute_trajectory_probs_single(ops:list[np.ndarray[np.complex128]], 
                                    state:np.ndarray[np.complex128], 
                                    qubit:int
                                )->np.ndarray[np.float64]:
    """
    Function to compute the probabilities of the noise operators for a single qubit gate.

    Parameters
    ----------
    ops : list[np.ndarray[np.complex128]]
        List of noise operators for the single qubit gate.
    state : np.ndarray[np.complex128]
        The current state of the qubit system after applying the gate.
    qubit : int
        The qubit to which the noise operators are applied.
    
    Returns
    -------
    np.ndarray[np.float64]
        The probabilities of the noise operators for the single qubit gate.
    """
    probs = np.zeros(len(ops), dtype=np.float64)
    for i in range(len(ops)):
        psi = get_updated_state_single(ops[i], state, qubit)
        probs[i] = np.vdot(psi, psi).real
    return probs

def get_updated_state_two_q(gate_op:np.ndarray[np.complex128], 
                            state:np.ndarray[np.complex128], 
                            q1:int, 
                            q2:int
                        )->np.ndarray[np.complex128]:
    """
    Function to get the updated state of the qubit system for a two qubit noise operator.

    Parameters
    ----------
    gate_op : np.ndarray[np.complex128]
        The noise operator to be applied to the qubit system.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q1 : int
        The first qubit to which the noise operator is applied.
    q2 : int
        The second qubit to which the noise operator is applied.
    
    Returns
    -------
    np.ndarray[np.complex128]
        The updated state of the qubit system after applying the two qubit noise operator.
    """
    psi_dash = np.zeros_like(state)
    dim = 1 << int(np.log2(state.size))
    iters = dim >> 2
    q_min = min(q1, q2)
    q_max = max(q1, q2)
    m1 = (1 << q_min) - 1
    m2 = (1 << (q_max - 1)) - 1
    ull_q1 = 1 << q1
    ull_q2 = 1 << q2
    target_mask = ull_q1 | ull_q2
    for i in range(iters):
        i_s1 = (i & m1) | ((i & ~m1) << 1)
        pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1)
        idx00 = pos
        idx01 = pos | ull_q2
        idx10 = pos | ull_q1
        idx11 = pos | target_mask
        psi_dash[idx00] += gate_op[0,0] * state[idx00] + gate_op[0,1] * state[idx01] + gate_op[0,2] * state[idx10] + gate_op[0,3] * state[idx11]
        psi_dash[idx01] += gate_op[1,0] * state[idx00] + gate_op[1,1] * state[idx01] + gate_op[1,2] * state[idx10] + gate_op[1,3] * state[idx11]
        psi_dash[idx10] += gate_op[2,0] * state[idx00] + gate_op[2,1] * state[idx01] + gate_op[2,2] * state[idx10] + gate_op[2,3] * state[idx11]
        psi_dash[idx11] += gate_op[3,0] * state[idx00] + gate_op[3,1] * state[idx01] + gate_op[3,2] * state[idx10] + gate_op[3,3] * state[idx11]
    return psi_dash

def compute_trajectory_probs_two_q(ops:list[np.ndarray[np.complex128]],
                                   state:np.ndarray[np.complex128], 
                                   qubits:list[int]
                                   )->np.ndarray[np.float64]:
    """
    Function to compute the probabilities of the noise operators for a two qubit gate.

    Parameters
    ----------
    ops : list[np.ndarray[np.complex128]]
        List of noise operators for the two qubit gate.
    state : np.ndarray[np.complex128]
        The current state of the qubit system after applying the gate.
    qubits : list[int]
        The two qubits to which the noise operators are applied.
    
    Returns
    -------
    np.ndarray[np.float64]
        The probabilities of the noise operators for the two qubit gate.
    """
    probs = np.zeros(len(ops), dtype=np.float64)
    for i in range(len(ops)):
        psi = get_updated_state_two_q(ops[i], state, qubits[0], qubits[1])
        probs[i] = np.vdot(psi, psi).real
    return probs

def update_state_inplace_1q(op:np.ndarray[np.complex128], 
                            state:np.ndarray[np.complex128], 
                            q:int
                            )->None:
    """
    Function that applies the single qubit noise operator the statevector inplace.

    Parameters
    ----------
    op : np.ndarray[np.complex128]
        The noise operator to be applied.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q : int
        The qubit to which the noise operator is applied to.
    
    Returns
    -------
    None
    """
    dim = 1 << int(np.log2(state.size))
    stride = 1 << q
    for p in range(0, dim >> 1, 1):
        i = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1)
        j = i | stride
        s0 = state[i]
        s1 = state[j]
        state[i] = op[0,0]*s0 + op[0,1]*s1
        state[j] = op[1,0]*s0 + op[1,1]*s1

def update_state_inplace_2q(op:np.ndarray[np.complex128],
                            state:np.ndarray[np.complex128],
                            q1:int,
                            q2:int
                            )->None:
    """
    Function that applies the two qubit noise operator to the statevector inplace.

    Parameters
    ----------
    op : np.ndarray[np.complex128]
        The noise operator to be applied.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q1 : int
        The first qubit to which the noise operator is applied to.
    q2 : int
        The second qubit to which the noise operator is applied to.
    
    Returns
    -------
    None
    """
    dim = 1 << int(np.log2(state.size))
    iters = dim >> 2
    q_min = min(q1, q2)
    q_max = max(q1, q2)
    m1 = (1 << q_min) - 1
    m2 = (1 << (q_max - 1)) - 1
    ull_q1 = 1 << q1
    ull_q2 = 1 << q2
    target_mask = ull_q1 | ull_q2
    for i in range(iters):
        i_s1 = (i & m1) | ((i & ~m1) << 1)
        pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1)
        idx00 = pos
        idx01 = pos | ull_q2
        idx10 = pos | ull_q1
        idx11 = pos | target_mask
        s00 = state[idx00]
        s01 = state[idx01]
        s10 = state[idx10]
        s11 = state[idx11]
        state[idx00] = op[0,0]*s00 + op[0,1]*s01 + op[0,2]*s10 + op[0,3]*s11
        state[idx01] = op[1,0]*s00 + op[1,1]*s01 + op[1,2]*s10 + op[1,3]*s11
        state[idx10] = op[2,0]*s00 + op[2,1]*s01 + op[2,2]*s10 + op[2,3]*s11
        state[idx11] = op[3,0]*s00 + op[3,1]*s01 + op[3,2]*s10 + op[3,3]*s11

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

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.
        single_qubit_noise : dict
            Dictionary containing the noise operators for single qubit gates.
        two_qubit_noise : dict)
            Dictionary containing the noise operators for two qubit gates.
        two_qubit_noise_index : dict
            Dictionary mapping qubit pairs to their indices in the two qubit noise dictionary.
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
            "rz": self._apply_single_qubit_noise,
            "rx": self._apply_single_qubit_noise,
            "cz": self._apply_two_qubit_noise,
            "ecr": self._apply_two_qubit_noise,
            "rzz": self._apply_single_qubit_noise,
            "unitary": self._no_noise
        }
    
    def _apply_single_qubit_noise(self,
                                   state:np.ndarray[np.complex128],
                                   gate_name:str,
                                   qubit_index:list[int]
                                   )->np.ndarray[np.complex128]:
        """
        Private method that applies applies the noise operator for single qubit gates.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate previously applied.
        qubit_index : list[int]
            List of qubits that the noise must be applied to.
        
        Returns
        -------
        np.ndarray[np.complex128]
            Updated statevector after applying the noise operator.
        """
        ops = self.single_qubit_noise[qubit_index[0]][1][gate_name]
        kraus_probs = compute_trajectory_probs_single(ops, state, qubit_index[0])
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        update_state_inplace_1q(ops[chosen_idx], state, qubit_index[0])
        return state / np.sqrt(kraus_probs[chosen_idx])
    
    def _apply_two_qubit_noise(self,
                               state:np.ndarray[np.complex128],
                               gate_name:str,
                               qubit_index:list[int]
                               )->np.ndarray[np.complex128]:
        """
        Private method that applies the noise operator for two qubit gates.

        Parameters
        ----------
        state : np.ndarray[np.complex128])
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate previously applied.
        qubit_index : list[int]
            List of qubits that the noise must be applied to.
        
        Returns
        -------
        np.ndarray[np.complex128]
            Updated statevector after applying the noise operator.
        """
        qubit_pair = tuple(qubit_index)
        ops = self.two_qubit_noise[self.two_qubit_noise_index[gate_name]][1][qubit_pair]
        kraus_probs = compute_trajectory_probs_two_q(ops, state, qubit_index)
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        update_state_inplace_2q(ops[chosen_idx], state, qubit_index[0], qubit_index[1])
        return state / np.sqrt(kraus_probs[chosen_idx])
    
    def _no_noise(self,
                  state:np.ndarray[np.complex128],
                  gate_name:str,
                  qubit_index:list[int]
                )->np.ndarray[np.complex128]:
        """
        Private method that returns the statevector unchanged when no noise is to be applied.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate previously applied.
        qubit_index : list[int]
            List of qubits that the noise would be applied to if there was noise.

        Returns
        -------
        np.ndarray[np.complex128]
            Unchanged statevector since no noise is applied.
        """
        return state

    def run(self,
            traj_id:int,
            instruction_list:list[list[str, list[int], float|None]]
        )->None:
        """
        Main method of the module to execute the MCWF trajectories.

        Parameters
        ----------
        traj_id : int
            Trajectory ID for the simulation.
        instruction_list : list[list[str, list[int], float|None]]
            List of instructions to build the quantum circuit.

        Returns
        -------
        None
        """
        self.instruction_list = instruction_list

        def compute_trajectory(
                traj_id:int
                )->np.ndarray[np.float64]:
            """
            Method to compute a single MCWF trajectory.

            Parameters
            ----------
            traj_id : int
                Trajectory ID for the simulation.

            Returns
            -------
            np.ndarray[np.float64]
                Probabilities after executing the trajectory.
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

    def get(self)->np.ndarray[np.float64]:
        """
        Method to get the accumulated probabilities after all trajectories have been run.

        Returns
        -------
        np.ndarray[np.float64]
            Accumulated probabilities after all trajectories.
        """
        return self.probs_sum
    
    def reset(self)->None:
        """
        Method to reset the accumulated probabilities and the measured qubits.
        """
        self.probs_sum = np.zeros(2**self.num_qubits, dtype=np.float64)
        