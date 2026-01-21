"""
This module allows users to create and simulate quantum circuits with noise models based on IBM quantum machines. It provides methods for adding gates, executing the circuit with Monte-Carlo simulations, and visualizing the circuit. It considers both single and two-qubit gate errors as well as measurement errors.\n

Example:\n
    >>> from NoisyCircuits.QuantumCircuit import QuantumCircuit
    >>> circuit = QuantumCircuit(num_qubits=3, noise_model=my_noise_model, backend_qpu_type='Heron', num_trajectories=1000)
    >>> circuit.h(0)
    >>> circuit.cx(0, 1)
    >>> circuit.cx(1, 2)
    >>> circuit.run_with_density_matrix(qubits=[0, 1, 2]) # Executes the circuit using the density matrix solver
    [0.39841323, 0.00300163, 0.09303931, 0.00615167, 0.00616272, 0.09281154, 0.00300024, 0.39741967]
    >>> circuit.execute(qubits=[0, 1, 2]) # Executes the circuit using the Monte-Carlo Wavefunction method
    [0.39748485, 0.0037614 , 0.09168292, 0.00799886, 0.00746056, 0.09156762, 0.00367236, 0.39637143]
    >>> circuit.run_pure_state(qubits=[0, 1, 2]) # Executes the circuit using the pure state solver
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    >>> circuit.shutdown() # Shutdown the Ray parallel execution environment
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from NoisyCircuits.utils.BuildQubitGateModel import BuildModel
from NoisyCircuits.utils.EagleDecomposition import EagleDecomposition
from NoisyCircuits.utils.HeronDecomposition import HeronDecomposition
from NoisyCircuits.utils.solvers import load_solver
from NoisyCircuits.utils.marginal_probs import compute_marginal_probs
import json
import ray
import gc


class QuantumCircuit:
    r"""
    This class allows a user to create a quantum circuit with error model from IBM machines where selected gates (both parameterized and non-parameterized) are implemented as methods. The gate decomposition uses the basis gates of the IBM Eagle (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)` and :math:`ECR`) / Heron (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)`, :math:`R_X(\theta)`, :math:`CZ` and :math:`RZZ(\theta)`) QPUs.

    Currently, it is only possible to apply a limited selection of single and two-qubit gates to the circuit simulation. For a full list of supported gates, please refer to the Decomposition :func:`NoisyCircuits.utils.Decomposition` class documentation. 

    Args:
        num_qubits (int): The number of qubits in the circuit.
        noise_model (dict): The noise model to be used for the circuit.
        backend_qpu_type (str): The IBM Backend Architecture type to be used (Eagle or Heron).
        num_trajectories (int): The number of trajectories for the Monte-Carlo simulation.
        num_cores (int, optional): The number of cores to use for parallel execution. Defaults to 2.
        jsonize (bool, optional): If True, the circuit will be serialized to JSON format. Defaults to False.
        verbose (bool, optional): If False, suppresses detailed output during initialization. Defaults to True.
        threshold (float, optional): The threshold for noise application. Defaults to 1e-12.

    Raises:
        TypeError: If num_qubits is not an integer.
        ValueError: If num_qubits is less than 1.
        TypeError: If the noise_model is not a dictionary.
        TypeError: If the backend_qpu_type is not a string.
        ValueError: If the backend_qpu_type is not one of ['Eagle', 'Heron'].
        TypeError: If num_trajectories is not an integer.
        ValueError: If num_trajectories is less than 1.
        TypeError: If num_cores is not an integer.
        ValueError: If num_cores is less than 1.
        ValueError: If num_cores exceeds the available CPU cores.
        TypeError: If sim_backend is not a string.
        ValueError: If sim_backend is not available.
        TypeError: If jsonize is not a boolean.
        TypeError: If verbose is not a boolean.
        TypeError: If threshold is not a float.
        ValueError: If threshold is not between 0 and 1 (exclusive).
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
    available_sim_backends = ["qulacs", "pennylane", "qiskit"]

    def __init__(self,
                 num_qubits:int,
                 noise_model:dict,
                 backend_qpu_type:str,
                 num_trajectories:int,
                 num_cores:int=2,
                 sim_backend:str="qulacs",
                 jsonize:bool=False,
                 verbose:bool=True,
                 threshold:float=1e-12)->None:
        """
        Initializes the QuantumCircuit with the specified number of qubits, noise model, number of trajectories for Monte-Carlo simulation, and threshold for noise application.
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
        if not isinstance(sim_backend, str):
            raise TypeError("Simulation backend must be a string.")
        if sim_backend.lower() not in QuantumCircuit.available_sim_backends:
            raise ValueError(f"Simulation backend must be one of {QuantumCircuit.available_sim_backends}.")
        if num_cores < 1:
            raise ValueError("Number of cores must be a positive integer greater than or equal to 1.")
        if not isinstance(jsonize, bool):
            raise TypeError("Jsonize must be a boolean.")
        if not isinstance(verbose, bool):
            raise TypeError("Verbose must be a boolean.")
        if num_cores > os.cpu_count():
            raise ValueError(f"Number of cores cannot exceed available CPU cores ({os.cpu_count()}).")
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        if jsonize:
            self.noise_model = json.JSONDecoder().decode(json.dumps(noise_model))
        self.num_trajectories = num_trajectories
        self.threshold = threshold
        self.num_cores = num_cores
        self._sim_backend = None
        self.solver = None
        self.sim_backend = sim_backend
        self.verbose = verbose
        self.qpu = backend_qpu_type.lower()
        basis_gates = QuantumCircuit.basis_gates_set[self.qpu]["basis_gates"]
        modeller = BuildModel(
                                noise_model=self.noise_model,
                                num_qubits=self.num_qubits,
                                threshold=self.threshold,
                                basis_gates=basis_gates,
                                verbose=self.verbose
                            )    
        single_error, multi_error, measure_error, connectivity = modeller.build_qubit_gate_model()
        self.single_qubit_instructions = single_error
        self.two_qubit_instructions = multi_error
        self.measurement_error = measure_error
        self.connectivity = connectivity
        single_qubit_instructions_array = np.array(list(self.single_qubit_instructions.items()))
        two_qubit_instructions_array = np.array(list(self.two_qubit_instructions.items()))
        self.qubit_coupling_map = modeller.qubit_coupling_map
        self.measurement_error_operator = self._generate_measurement_error_operator()
        self._gate_decomposer = QuantumCircuit.basis_gates_set[self.qpu]["gate_decomposition"](
                                                                                                num_qubits=self.num_qubits,
                                                                                                connectivity=self.connectivity,
                                                                                                qubit_map=self.qubit_coupling_map
                                                                                            )
        ray.init(num_cpus=self.num_cores, ignore_reinit_error=True, log_to_driver=False)
        self._single_qubit_instruction_reference = ray.put(single_qubit_instructions_array)
        self._two_qubits_instruction_reference = ray.put(two_qubit_instructions_array)
        self._two_qubit_gate_index = {two_qubit_instructions_array[i][0] : i for i in range(len(two_qubit_instructions_array))}
        self.workers = [
                    self.solver.RemoteExecutor.remote(
                                        num_qubits = self.num_qubits,
                                        single_qubit_noise = self._single_qubit_instruction_reference,
                                        two_qubit_noise = self._two_qubits_instruction_reference,
                                        two_qubit_noise_index = self._two_qubit_gate_index
                            ) for _ in range(self.num_cores)
        ]

    @property
    def sim_backend(self)->str:
        """
        Getter for the _sim_backend attribute

        Returns:
            (str): Returns the current sim_backend value.
        """
        return self._sim_backend
    
    @sim_backend.setter
    def sim_backend(self,
                    backend:str)->None:
        """
        Setter for the _sim_backend attribute and updates the solver modules.

        Args:
            backend (str): The name of the backend.
        
        Raises:
            TypeError: Raised when backend is not a string
            ValueError: Raised when the specified backend is not available.
        """
        if not isinstance(backend, str):
            raise TypeError("Specified backend must of type string")
        if backend not in QuantumCircuit.available_sim_backends:
            raise ValueError(f"Specified backend {backend} is not available. Choose from {QuantumCircuit.available_sim_backends}.")
        if backend == self._sim_backend:
            print("Backend already in use.")
            return
        new_solver = load_solver(backend)
        self.solver = new_solver
        self._sim_backend = backend
        print("Successfully switched backend to {}.".format(backend))

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
    
    def _generate_measurement_error_operator(self,
                                            qubit_list:list[int]=None)->np.ndarray:
        """
        Generates the measurement error operator for the specified qubits.
        
        Args:
            qubit_list (list[int], optional): The list of qubits to include in the measurement error operator. If None, includes all qubits. Defaults to None.

        Returns:
            np.ndarray: The measurement error operator as a numpy array. Returns None if there are no measurement errors.
        """
        if qubit_list is None:
            measure_qubits = list(range(self.num_qubits))
        else:
            measure_qubits = qubit_list
        if self.measurement_error == {}:
            return None
        for qubit_number, qubit in enumerate(measure_qubits):
            if qubit_number == 0:
                meas_error_op = self.measurement_error[qubit]
            else:
                meas_error_op = np.kron(meas_error_op, self.measurement_error[qubit])
        return meas_error_op

    def execute(self,
                qubits:list[int],
                num_trajectories:int=None)->np.ndarray:
        """
        Executes the built quantum circuit with the specified noise model using the Monte-Carlo Wavefunction method.

        Args:
            qubits (list[int]): The list of qubits to be measured.
            num_trajectories (int): The number of trajectories for the Monte-Carlo simulation (can be modified). Defaults to None and uses the class attribute. If specified, it overrides the class attribute for only this execution. Defaults to None.
        
        Raises:
            TypeError: If qubits is not a list or the items in the list are not integers.
            TypeError: If num_trajectories is not an integer.
            ValueError: If num_trajectories is less than 1.
            ValueError: If qubits contains invalid qubit indices.
            ValueError: If there are no instructions in the circuit to execute.

        Returns:
            np.ndarray: The probabilities of the output states.
        """
        if num_trajectories is not None:
            if not isinstance(num_trajectories, int):
                raise TypeError("Number of trajectories must be an integer.")
            if num_trajectories < 1:
                raise ValueError("Number of trajectories must be a positive integer greater than or equal to 1.")
        if not isinstance(qubits, list) or any(not isinstance(qubit, int) for qubit in qubits):
            raise TypeError("qubits must be of type list.\nAll entries in qubits must be integers.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        if len(qubits) != self.num_qubits:
            measurement_error_operator = self._generate_measurement_error_operator(qubit_list=qubits)
        else:
            measurement_error_operator = self.measurement_error_operator
        
        reset_probs = [
            self.workers[i].reset.remote(measured_qubits=qubits) for i in range(self.num_cores)
        ]

        futures = [
            self.workers[traj_id % self.num_cores].run.remote(traj_id, self.instruction_list) for traj_id in range(num_trajectories)
            ]
        
        prob_chunks = [
            ray.get(self.workers[i].get.remote(qubits)) for i in range(self.num_cores)
        ]

        probs = np.array(prob_chunks).sum(axis=0) / num_trajectories

        if self._sim_backend not in ["pennylane"]:
            probs = probs.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)

        if len(qubits) != self.num_qubits and self._sim_backend not in ["pennylane"]:
            trace_qubits = [i for i in range(self.num_qubits) if i not in qubits]
            probs_reduced = compute_marginal_probs(probs, trace_qubits)
            probs = probs_reduced
        
        if measurement_error_operator is not None:
            probs = np.dot(measurement_error_operator, probs)
        return probs

    def run_with_density_matrix(self, 
                                qubits:list[int])->np.ndarray:
        """
        Runs the quantum circuit with the density matrix solver.

        Args:
            qubits (list[int]): List of qubits to be simulated.

        Raises:
            TypeError: If qubits is not a list or the items in the list are not integers.
            ValueError: If qubits contains invalid qubit indices.
            ValueError: If there are no instructions in the circuit to execute.

        Returns:
            np.ndarray: Probabilities of the output states.
        """
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if len(qubits) != self.num_qubits:
            measurement_error_operator = self._generate_measurement_error_operator(qubit_list=qubits)
        else:
            measurement_error_operator = self.measurement_error_operator
        density_matrix_solver = self.solver.DensityMatrixSolver(
                    num_qubits=self.num_qubits,
                    single_qubit_noise=self.single_qubit_instructions,
                    two_qubit_noise=self.two_qubit_instructions,
                    instruction_list=self.instruction_list
                )
        probs = density_matrix_solver.solve(qubits=qubits)

        if self._sim_backend not in ["pennylane"]:
            m = len(qubits)
            probs = probs.reshape([2]*m).transpose(list(range(m))[::-1]).reshape(-1)
        
        if measurement_error_operator is not None:
            probs = np.dot(measurement_error_operator, probs)
        return probs

    def run_pure_state(self, 
                       qubits:list[int])->np.ndarray:
        """
        Runs the quantum circuit with the pure state solver.

        Args:
            qubits (list[int]): List of qubits to be simulated.
        
        Raises:
            TypeError: If qubits is not a list or the items in the list are not integers.
            ValueError: If qubits contains invalid qubit indices.
            ValueError: If there are no instructions in the circuit to execute.

        Returns:
            np.ndarray: Probabilities of the output states.
        """
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        pure_state_solver = self.solver.PureStateSolver(
                    num_qubits=self.num_qubits,
                    instruction_list=self.instruction_list
                    )
        probs = pure_state_solver.solve(qubits=qubits)
        if self._sim_backend not in ["pennylane"]:
            probs = probs.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)
        if len(qubits) != self.num_qubits and self._sim_backend not in ["pennylane"]:
            trace_qubits = [i for i in range(self.num_qubits) if i not in qubits]
            probs_reduced = compute_marginal_probs(probs, trace_qubits)
            probs = probs_reduced
        return probs
    
    def draw_circuit(self,
                     style:str="mpl")->None:
        """
        Draws the quantum circuit.

        Args:
            style (str, optional): The style of the drawing, either 'mpl' for matplotlib or 'text' for text-based representation. Defaults to 'mpl'.
        
        Raises:
            TypeError: If style is not a string.
            ValueError: If style is not one of ['mpl', 'text'].
            ValueError: If there are no instructions in the circuit to draw.
        """
        if not isinstance(style, str):
            raise TypeError("Style must be a string.")
        if style.lower() not in ["mpl", "text"]:
            raise ValueError("Style must be one of ['mpl', 'text'].")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to draw.")
        from qiskit import QuantumCircuit as QiskitQuantumCircuit
        circuit = QiskitQuantumCircuit(self.num_qubits)
        instruction_map = {
            "x": lambda q, p: circuit.x(q[0]),
            "sx": lambda q, p: circuit.sx(q[0]),
            "rz": lambda q, p: circuit.rz(p, q[0]),
            "rx": lambda q, p: circuit.rx(p, q[0]),
            "cz": lambda q, p: circuit.cz(q[0], q[1]),
            "ecr": lambda q, p: circuit.ecr(q[0], q[1]),
            "rzz": lambda q, p: circuit.rzz(p, q[0], q[1]),
            "unitary": lambda q, p: circuit.unitary(p, q)
        }
        for gate_name, qubit_index, parameters in self.instruction_list:
            instruction_map[gate_name](qubit_index, parameters)
        if style.lower() == "mpl":
            circuit.draw(output="mpl")
        else:
            circuit.draw()
        del circuit
        gc.collect()
    
    def shutdown(self):
        """
        Shutsdown the Ray parallel execution environment.
        """
        ray.shutdown()