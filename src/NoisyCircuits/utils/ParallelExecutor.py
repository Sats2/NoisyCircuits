"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel (shared/distributed memory) environment.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pennylane as qml
from pennylane import numpy as np
import scipy.sparse
import ray

@ray.remote
class RemoteExecutor:
    """
    Module that performs the parallel execution of the quantum circuit according to the Monte-Carlo Wavefunction method.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict)->None:
        """
        Constructor for the Remote Executor class.

        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            single_qubit_noise (dict): Dictionary containing the noise operators for single qubit gates.
            two_qubit_noise (dict): Dictionary containing the noise operators for two qubit gates.
        """
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = two_qubit_noise
        
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
        def apply_gate_noparams(state:np.ndarray, 
                                gate_op:callable, 
                                qubit_list:list)->np.ndarray:
            """
            Apply Non-parameteric gates to the quantum circuit.

            Args:
                state (np.ndarray): Current state of the qubit system.
                gate_op (callable): Operator to apply on the qubit system.
                qubit_list (list): List of qubits that the operator must bbe applied to.
            
            Returns:
                np.ndarray: The updated state of the qubit system.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            gate_op(qubit_list)
            return qml.state()
            
        @qml.qnode(device=self.dev)
        def apply_gate_params(state:np.ndarray, 
                              gate_op:callable, 
                              params:np.ndarray, 
                              qubit_list:list):
            """
            Apply Parameteric gates to the quantum circuit

            Args:
                state (np.ndarray): Current state of the qubit system.
                gate_op (callable): Operator to apply on the qubit system.
                params (np.ndarray): The value for the parameterized gate.
                qubit_list (list): List of qubits that the operator must bbe applied to.

            Returns:
                np.ndarray: The updated state of the qubit system.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            gate_op(params, qubit_list)
            return qml.state()
            
        @qml.qnode(device=self.dev)
        def get_probs(state:np.ndarray)->np.ndarray:
            """
            Compute the final probabilties of the qubit system in the trajectory.

            Args:
                state (np.ndarray): Final state of the qubit system in the current trajectory.

            Returns:
                np.ndarray: Probabilities of the measured qubits from the qubit system.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            return qml.probs(wires=self.measured_qubits)
            
        self.apply_gate_noparams = apply_gate_noparams
        self.apply_gate_params = apply_gate_params
        self.get_probs = get_probs

    def _create_gate_handlers(self):
        """Create gate handler functions to eliminate conditionals in main loop"""
        
        def safe_apply_gate_noparams(state:np.ndarray, 
                                     gate_op:callable, 
                                     qubits:list)->np.ndarray:
            """
            Apply gate with NaN handling only when needed.

            Args:
                state (np.ndarray): Current state of the qubit system.
                gate_op (callable): Operator to apply on the qubit system.
                qubits (list): List of qubits the the operator is applied to.

            Returns:
                np.ndarray: Updated state afer NaN checks.
            """
            psi_dash = self.apply_gate_noparams(state, gate_op, qubits)
            if np.isnan(psi_dash).any():
                noisy_state = state + np.random.normal(0, 1e-8, size=state.shape)
                psi_dash = self.apply_gate_noparams(noisy_state, gate_op, qubits)
            return psi_dash
        
        def handle_two_qubit_gates(state:np.ndarray, 
                                   gate:str, 
                                   qubits:list, 
                                   params:np.ndarray)->np.ndarray:
            """
            Apply two qubit gate with NaN handling when needed.

            Args:
                state (np.ndarray): Current state of the qubit system.
                gate (str): Name of the gate to be applied to the system.
                qubits (list): List of qubits that the gate must be applied to.
                params (np.ndarray): Parameter value for parameterized two qubit gates.

            Returns:
                np.ndarray: Updated state after NaN checks.
            """
            qpair = tuple(qubits)
            psi_dash = safe_apply_gate_noparams(state, self.instruction_map[gate], qubits)
            ops = self.two_qubit_noise[gate][qpair]["operators"]
            op_psi = np.array([op.dot(psi_dash) for op in ops])
            kraus_probs = np.real([np.vdot(psi, psi) for psi in op_psi])
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            prob_sqrt = np.sqrt(kraus_probs[kraus_idx])
            return ops[kraus_idx].dot(psi_dash) / prob_sqrt if prob_sqrt > 1e-12 else ops[kraus_idx].dot(psi_dash)
            
        def handle_param_gate(state:np.ndarray, 
                              gate:str, 
                              qubits:list, 
                              params:np.ndarray)->np.ndarray:
            """
            Apply parameterized gates with NaN handling.

            Args:
                state (np.ndarray): Current state of the qubit system.
                gate (str): Name of the gate to be applied to the qubit system.
                qubits (list): List of qubits to applied the gate to.
                params (np.ndarray): Parameter values to apply.
            
            Returns:
                np.ndarray: Updated state of the system after NaN checks.
            """
            result = self.apply_gate_params(state, self.instruction_map[gate], params, qubits)
            if np.isnan(result).any():
                noisy_state = state + np.random.normal(0, 1e-8, size=state.shape)
                result = self.apply_gate_params(noisy_state, self.instruction_map[gate], params, qubits)
            return result
            
        def handle_single_qubit_noise(state:np.ndarray, 
                                      gate:str, 
                                      qubits:list, 
                                      params:np.ndarray)->np.ndarray:
            """
            Applies the noise operator to the system for single qubit gates.

            Args:
                state (np.ndarray): Current State of the qubit system.
                gate (str): Name of the gate to apply.
                qubits (list): Qubit to which the gate is applied.
                params (np.ndarray): Value for the parameterized gate.

            Returns:
                np.ndarray: Updated state of the system after noise application.
            """
            psi_dash = safe_apply_gate_noparams(state, self.instruction_map[gate], qubits)
            ops = self.single_qubit_noise[qubits[0]][gate]["kraus_operators"]
            op_psi = np.array([op.dot(psi_dash) for op in ops])
            kraus_probs = np.real([np.vdot(psi, psi) for psi in op_psi])
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            prob_sqrt = np.sqrt(kraus_probs[kraus_idx])
            return ops[kraus_idx].dot(psi_dash) / prob_sqrt if prob_sqrt > 1e-12 else ops[kraus_idx].dot(psi_dash)
        
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
            instruction_list:list,
            measured_qubits:list[int])->np.ndarray:
        """
        Main method of the module to execute the MCWF trajectory.

        Args:
            traj_id (int): Trajectory ID.
            instruction_list (list): List of instructions to build the quantum circuit.
            measured_qubits (list[int]): List of qubits to measure.
        
        Returns:
            np.ndarray: Probabilities of the measured qubits after the circuit execution.
        """
        self.instruction_list = instruction_list
        self.measured_qubits = measured_qubits
        
        def compute_trajectory(traj_id:int)->np.ndarray:
            """
            Computes the probabilities of the current MCWF trajectory.

            Args:
                traj_id (int): Trajectory ID.
            
            Returns:
                np.ndarray: Probabilities from the current trajectory.
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
            return np.zeros(probs.shape) if np.isnan(probs).any() else probs
        
        return compute_trajectory(traj_id)