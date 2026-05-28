"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using pennylane as a quantum circuit simulator backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel (shared/distributed memory) environment.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pennylane as qml
import numpy as np
import ray
from numba import njit


def get_updated_state_single(gate_op:np.ndarray[np.complex128], 
                             state:np.ndarray[np.complex128], 
                             q:int)->np.ndarray[np.complex128]:
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
                                    qubit:int)->np.ndarray[np.float64]:
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
                            q2:int)->np.ndarray[np.complex128]:
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

def compute_trajectory_probs_two_q(ops:list[np.ndarray[np.complex128]], state:np.ndarray[np.complex128], qubits:list[int])->np.ndarray[np.float64]:
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
                            q:int)->None:
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
                            q2:int)->None:
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
        self.measured_qubits = num_qubits
        self.probs_sum = np.zeros(2**self.measured_qubits, dtype=np.float64)
        
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
        @qml.qnode(device=self.dev, interface=None)
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
            
        @qml.qnode(device=self.dev, interface=None)
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
            
        @qml.qnode(device=self.dev, interface=None)
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
            ops = self.two_qubit_noise[self.two_qubit_noise_index[gate]][1][qpair]["qubit_channel"]
            kraus_probs = compute_trajectory_probs_two_q(ops, psi_dash, qubits)
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            del kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            update_state_inplace_2q(ops[kraus_idx], psi_dash, qubits[0], qubits[1])
            return psi_dash / np.sqrt(kraus_probs[kraus_idx])
            
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
            ops = self.single_qubit_noise[qubits[0]][1][gate]["qubit_channel"]
            kraus_probs = compute_trajectory_probs_single(ops, psi_dash, qubits[0])
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            del kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            update_state_inplace_1q(ops[kraus_idx], psi_dash, qubits[0])
            return psi_dash / np.sqrt(kraus_probs[kraus_idx])
        
        self.gate_handlers = {}
        for gate in self.instruction_map:
            if gate in ["ecr", "cz"]:
                self.gate_handlers[gate] = handle_two_qubit_gates
            elif gate in self.param_gates:
                self.gate_handlers[gate] = handle_param_gate
            else:
                self.gate_handlers[gate] = handle_single_qubit_noise

    def run(self, 
            traj_id:int,
            instruction_list:list)->np.ndarray[np.float64]:
        """
        Main method of the module to execute the MCWF trajectory.

        Args:
            traj_id (int): Total number of trajectories that need to be run by the core.
            instruction_list (list): List of instructions to build the quantum circuit.
            measured_qubits (list[int]): List of qubits to measure.
        
        Returns:
            np.ndarray[np.float64]: Probabilities of the measured qubits after the circuit execution.
        """
        self.instruction_list = instruction_list
        
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
                state = self.gate_handlers[gate](state, gate, qubits, params)
                if np.isnan(state).any():
                    return np.zeros(2**len(self.measured_qubits))
            
            probs = self.get_probs(state)
            del state
            return probs
        
        self.probs_sum += compute_trajectory(traj_id)

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
        self.measured_qubits = measured_qubits
        self.probs_sum = np.zeros(2**len(self.measured_qubits), dtype=np.float64)