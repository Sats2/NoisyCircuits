import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pennylane as qml
from pennylane import numpy as np
import numpy as onp
import ray

@ray.remote
class RemoteExecutor:
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict):
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
        def apply_gate_noparams(state, gate_op, qubit_list):
            qml.StatePrep(state, wires=range(self.num_qubits))
            gate_op(qubit_list)
            return qml.state()
            
        @qml.qnode(device=self.dev)
        def apply_gate_params(state, gate_op, params, qubit_list):
            qml.StatePrep(state, wires=range(self.num_qubits))
            gate_op(params, qubit_list)
            return qml.state()
            
        @qml.qnode(device=self.dev)
        def get_probs(state):
            qml.StatePrep(state, wires=range(self.num_qubits))
            return qml.probs(wires=self.measured_qubits)
            
        self.apply_gate_noparams = apply_gate_noparams
        self.apply_gate_params = apply_gate_params
        self.get_probs = get_probs

    def _create_gate_handlers(self):
        """Create gate handler functions to eliminate conditionals in main loop"""
        
        def safe_apply_gate_noparams(state, gate_op, qubits):
            """Apply gate with NaN handling only when needed"""
            psi_dash = self.apply_gate_noparams(state, gate_op, qubits)
            if np.isnan(psi_dash).any():
                noisy_state = state + np.random.normal(0, 1e-8, size=state.shape)
                psi_dash = self.apply_gate_noparams(noisy_state, gate_op, qubits)
            return psi_dash
        
        def handle_two_qubit_gates(state, gate, qubits, params):
            qpair = tuple(qubits)
            psi_dash = safe_apply_gate_noparams(state, self.instruction_map[gate], qubits)
            ops = self.two_qubit_noise[gate][qpair]["operators"]
            op_psi = np.array([op @ psi_dash for op in ops])
            kraus_probs = np.real(np.sum(np.conj(op_psi) * op_psi, axis=1))
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            prob_sqrt = np.sqrt(kraus_probs[kraus_idx])
            return ops[kraus_idx] @ psi_dash / prob_sqrt if prob_sqrt > 1e-12 else ops[kraus_idx] @ psi_dash
            
        def handle_param_gate(state, gate, qubits, params):
            result = self.apply_gate_params(state, self.instruction_map[gate], params, qubits)
            if np.isnan(result).any():
                noisy_state = state + np.random.normal(0, 1e-8, size=state.shape)
                result = self.apply_gate_params(noisy_state, self.instruction_map[gate], params, qubits)
            return result
            
        def handle_single_qubit_noise(state, gate, qubits, params):
            psi_dash = safe_apply_gate_noparams(state, self.instruction_map[gate], qubits)
            ops = self.single_qubit_noise[qubits[0]][gate]["kraus_operators"]
            op_psi = np.array([op @ psi_dash for op in ops])
            kraus_probs = np.real(np.sum(np.conj(op_psi) * op_psi, axis=1))
            kraus_probs_sum = np.sum(kraus_probs)
            if kraus_probs_sum == 0 or np.isnan(kraus_probs_sum):
                kraus_probs = np.ones(len(kraus_probs)) / len(kraus_probs)
            else:
                kraus_probs /= kraus_probs_sum
            kraus_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
            prob_sqrt = np.sqrt(kraus_probs[kraus_idx])
            return ops[kraus_idx] @ psi_dash / prob_sqrt if prob_sqrt > 1e-12 else ops[kraus_idx] @ psi_dash
        
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
            measured_qubits:list[int]):
        self.instruction_list = instruction_list
        self.measured_qubits = measured_qubits
        
        def compute_trajectory(traj_id):
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
