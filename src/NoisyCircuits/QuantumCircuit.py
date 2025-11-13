import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pennylane import numpy as np
from NoisyCircuits.utils.BuildQubitGateModel import BuildModel
from NoisyCircuits.utils.DensityMatrixSolver import DensityMatrixSolver
from NoisyCircuits.utils.PureStateSolver import PureStateSolver
from NoisyCircuits.utils.ParallelExecutor import RemoteExecutor
from NoisyCircuits.utils.EagleDecomposition import EagleDecomposition
from NoisyCircuits.utils.HeronDecomposition import HeronDecomposition
import json
import ray


class QuantumCircuit:
    """
    This class allows a user to create a quantum circuit with error model from IBM machines where selected gates (both parameterized and non-parameterized) are 
    implemented as methods. The gate decomposition uses RZ, SX and X gates for single qubit operations and ECR gate for two qubit operations as the basis gates.
    """

    # Update QPU Basis Gates Here!
    basis_gates_set = {
        'eagle': {
                    "basis_gates" : [['rz', 'sx', 'x'], ['ecr']],
                    "gate_decomposition" : EagleDecomposition
                },
        'heron': {
                    "basis_gates" : [['rz', 'rx', 'sx', 'x'], ['cz', 'rzz']],
                    "gate_decomposition" : HeronDecomposition
                }
    }

    def __init__(self,
                 num_qubits:int,
                 noise_model:dict,
                 backend_qpu_type:str,
                 num_trajectories:int,
                 num_cores:int=2,
                 jsonize:bool=False,
                 threshold:float=1e-12)->None:
        """
        Initializes the QuantumCircuit with the specified number of qubits, noise model, number of trajectories for Monte-Carlo simulation, and threshold for noise application.

        Args:
            num_qubits (int): The number of qubits in the circuit.
            noise_model (dict): The noise model to be used for the circuit.
            backend_qpu_type (str): The IBM Backend Architecture type to be used (Eagle or Heron).
            num_trajectories (int): The number of trajectories for the Monte-Carlo simulation.
            num_cores (int, optional): The number of cores to use for parallel execution. Defaults to 2.
            jsonize (bool, optional): If True, the circuit will be serialized to JSON format. Defaults to False.
            threshold (float, optional): The threshold for noise application. Defaults to 1e-12.

        Raises:
            TypeError: If num_qubits is not an integer.
            ValueError: If num_qubits is less than 1.
            TypeError: If the noise_model is not a dictionary.
            TypeError: If the backend_qpu_type is not a string.
            ValueError: If the backend_qpu_type is not one of ['Eagle', 'Heron'].
            TypeError: If num_trajectories is not an integer.
            ValueError: If num_trajectories is less than 1.
            TypeError: If threshold is not a float.
            ValueError: If threshold is not between 0 and 1 (exclusive).
            TypeError: If num_cores is not an integer.
            ValueError: If num_cores is less than 1.
            TypeError: If jsonize is not a boolean.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("Number of qubits must be an integer.")
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        if not isinstance(noise_model, dict):
            raise TypeError("Noise model must be a dictionary.")
        if not isinstance(backend_qpu_type, str):
            raise TypeError("Backend QPU type must be a string.")
        if backend_qpu_type.lower() not in list(QuantumCircuit.basis_gates_set.keys()):
            raise ValueError(f"Backend QPU type must be one of {list(QuantumCircuit.basis_gates_set.keys())}.")
        if not isinstance(num_trajectories, int):
            raise TypeError("Number of trajectories must be an integer.")
        if num_trajectories < 1:
            raise ValueError("Number of trajectories must be a positive integer greater than or equal to 1.")
        if not isinstance(threshold, float):
            raise TypeError("Threshold must be a float.")
        if 0 >= threshold or threshold >= 1:
            raise ValueError("Threshold must be a float between 0 and 1 (exclusive).")
        if not isinstance(num_cores, int):
            raise TypeError("Number of cores must be an integer.")
        if num_cores < 1:
            raise ValueError("Number of cores must be a positive integer greater than or equal to 1.")
        if not isinstance(jsonize, bool):
            raise TypeError("Jsonize must be a boolean.")
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        if jsonize:
            self.noise_model = json.JSONDecoder().decode(json.dumps(noise_model))
        self.num_trajectories = num_trajectories
        self.threshold = threshold
        self.num_cores = num_cores
        basis_gates = QuantumCircuit.basis_gates_set[backend_qpu_type.lower()]["basis_gates"]
        modeller = BuildModel(
                                noise_model=self.noise_model,
                                num_qubits=self.num_qubits,
                                threshold=self.threshold,
                                basis_gates=basis_gates
                            )    
        single_error, multi_error, measure_error, connectivity = modeller.build_qubit_gate_model()
        self.single_qubit_instructions = single_error
        self.ecr_error_instruction = multi_error
        self.measurement_error = measure_error
        self.connectivity = connectivity
        self.qubit_coupling_map = modeller.qubit_coupling_map
        self.measurement_error_operator = None
        self._gate_decomposer = QuantumCircuit.basis_gates_set[backend_qpu_type.lower()]["gate_decomposition"](
                                                                                                                num_qubits=self.num_qubits,
                                                                                                                connectivity=self.connectivity,
                                                                                                                qubit_map=self.qubit_coupling_map
                                                                                                            )
        ray.init(num_cpus=self.num_cores, ignore_reinit_error=True, log_to_driver=False)
        self.workers = [
            RemoteExecutor.remote(
                num_qubits=self.num_qubits,
                single_qubit_noise=self.single_qubit_instructions,
                ecr_dict=self.ecr_error_instruction
            ) for _ in range(self.num_cores)]

    def __getattr__(self, name: str) -> callable:
        """
        Delegate unknown attributes/methods to the selected methods class.
        """
        if name is not None:
            return getattr(self._gate_decomposer, name)

    def refresh(self):
        """
        Resets the quantum circuit by clearing the instruction list and qubit-to-instruction mapping.
        """
        self._gate_decomposer.instruction_list = []
    
    def generate_measurement_operator(self,
                                      qubits:list[int])->np.ndarray:
        """
        Generates the measurement error operator for the specified qubits based on the noise model.

        Args:
            qubits (list[int]): List of qubit indices to include in the measurement error operator.

        Returns:
            np.ndarray: The measurement error operator as a NumPy array.
        """
        for q_num, qubit in enumerate(qubits):
            if q_num == 0:
                measure_op = self.measurement_error[qubit]
            else:
                measure_op = np.kron(measure_op, self.measurement_error[qubit])
        return measure_op
    
    def instantiate_measurement(self,
                            qubits:list[int])->None:
        if not isinstance(qubits, (list, range)):
            raise TypeError("Qubits must be a list of integers.")
        if isinstance(qubits, range):
            qubits = list(qubits)
        if not all(isinstance(q, int) for q in qubits):
            raise TypeError("All qubits must be integers.")
        if not all(0 <= q < self.num_qubits for q in qubits):
            raise ValueError(f"Qubits must be in the range [0, {self.num_qubits - 1}].")
        print(f"Creating Measurement Error Operator for observable qubits: {qubits}")
        self.measurement_error_operator = self.generate_measurement_operator(qubits)
        print(f"Measurement Error Operator created.")

    def execute(self,
                qubits:list[int],
                num_trajectories:int=None,
                use_prev:bool=True)->np.ndarray:
        """
        Executes the built quantum circuit with the specified noise model using the Monte-Carlo Wavefunction method.

        Args:
            qubits (list[int]): The list of qubits to be measured.
            num_trajectories (int): The number of trajectories for the Monte-Carlo simulation (can be modified). Defaults to None and uses the class attribute.
                                    If specified, it overrides the class attribute for this execution. Defaults to None.
        
        Raises:
            TypeError: If qubits is not a list.
            TypeError: If num_trajectories is not an integer.
            ValueError: If num_trajectories is less than 1.
            ValueError: If qubits contains invalid qubit indices.

        Returns:
            np.ndarray: The probabilities of the output states.
        """
        if use_prev:
            self.num_trajectories = num_trajectories if isinstance(num_trajectories, int) else self.num_trajectories
        else:
            self.instantiate_measurement(qubits)
        futures = [self.workers[traj_id % self.num_cores].run.remote(traj_id, self.instruction_list, qubits) for traj_id in range(self.num_trajectories)]
        probs_raw = np.array(ray.get(futures))
        probs = np.mean(probs_raw, axis=0)
        if self.measurement_error_operator is not None:
            probs = np.dot(self.measurement_error_operator, probs)
        return probs

    def run_with_density_matrix(self, 
                                qubits:list[int])->np.ndarray:
        """
        Runs the quantum circuit with the density matrix solver.

        Args:
            qubits (list[int]): List of qubits to be simulated.

        Returns:
            np.ndarray: Probabilities of the output states.
        """
        print(f"Creating Measurement Error Operator for observable qubits: {qubits}")
        measurement_error_operator =self.generate_measurement_operator(qubits)
        print(f"Measurement Error Operator created.\nExecuting the circuit with Density Matrix.")
        density_matrix_solver = DensityMatrixSolver(
            num_qubits=self.num_qubits,
            single_qubit_noise=self.single_qubit_instructions,
            ecr_noise=self.ecr_error_instruction,
            measurement_noise=self.measurement_error,
            instruction_list=self.instruction_list
        )
        probs = density_matrix_solver.solve(qubits=qubits)
        if measurement_error_operator is not None:
            probs = np.dot(measurement_error_operator, probs)
        return probs

    def run_pure_state(self, 
                       qubits:list[int])->np.ndarray:
        """
        Runs the quantum circuit with the pure state solver.

        Args:
            qubits (list[int]): List of qubits to be simulated.

        Returns:
            np.ndarray: Probabilities of the output states.
        """
        pure_state_solver = PureStateSolver(
            num_qubits=self.num_qubits,
            instruction_list=self.instruction_list
            )
        return pure_state_solver.solve(qubits=qubits)
    
    def shutdown(self):
        ray.shutdown()