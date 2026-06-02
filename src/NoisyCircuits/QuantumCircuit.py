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
from NoisyCircuits.utils import compute_marginal_probs, convert_matrix_to_little_endian
import measurement_error_applicator
import ray
import gc
import os
from collections.abc import Callable


class QuantumCircuit:
    r"""
    This class allows a user to create a quantum circuit with error model from IBM machines where selected gates (both parameterized and non-parameterized) are implemented as methods. The gate decomposition uses the basis gates of the IBM Eagle (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)` and :math:`ECR`) / Heron (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)`, :math:`R_X(\theta)`, :math:`CZ` and :math:`RZZ(\theta)`) QPUs.

    Currently, it is only possible to apply a limited selection of single and two-qubit gates to the circuit simulation. For a full list of supported gates, please refer to the Decomposition :func:`NoisyCircuits.utils.Decomposition` class documentation.

    Parameters
    -----------
    num_qubits : int
        The number of qubits in the quantum circuit.
    noise_model : dict
        A dictionary containing the raw noise model for the quantum circuit.
    backend_qpu_type : str
        The QPU architecture to use. Supported options are eagle and heron. (Defaults to heron)
    sim_backend : str
        The simulation backend to use for propogating the quantum circuit. Supported options are custom, pennylane, qiskit and qulacs. (Defaults to custom)
    threshold : float
        The threshold for pruning errors in the noise model. (Defaults to 1e-12)
    verbose : bool
        Whether to print verbose output during the noise model construction. (Defaults to True)

    Raises
    -------
    TypeError : 
        - num_qubits must be an integer.
        - noise_model must be a dictionary.
        - backend_qpu_type must be a string.
        - sim_backend must be a string.
        - threshold must be a float.
        - verbose must be a boolean.
    ValueError :
        - num_qubits must be a positive integer.
        - backend_qpu_type must be one of the supported QPU architectures.
        - sim_backend must be one of the supported simulation backends.
        - threshold must be between 0 and 1 (exclusive).
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
    available_sim_backends = ["custom", "pennylane", "qiskit", "qulacs"]

    def __init__(self, 
                 num_qubits:int,
                 noise_model:dict,
                 backend_qpu_type:str="heron",
                 sim_backend:str="custom",
                 threshold:float=1e-12,
                 verbose:bool=True
                )->None:
        """
        Initializes the QuantumCircuit with the specified number of qubits, noise model, number of trajectories for Monte-Carlo simulation, and threshold for noise application.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer.")
        if num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer.")
        if not isinstance(noise_model, dict):
            raise TypeError("noise_model must be a dictionary.")
        if not isinstance(backend_qpu_type, str):
            raise TypeError("backend_qpu_type must be a string.")
        if backend_qpu_type.lower() not in QuantumCircuit.basis_gates_set:
            raise ValueError(f"backend_qpu_type must be one of {list(QuantumCircuit.basis_gates_set.keys())}.")
        if not isinstance(sim_backend, str):
            raise TypeError("sim_backend must be a string.")
        if sim_backend.lower() not in QuantumCircuit.available_sim_backends:
            raise ValueError(f"sim_backend must be one of {QuantumCircuit.available_sim_backends}.")
        if not isinstance(threshold, float):
            raise TypeError("threshold must be a float.")
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be between 0 and 1 (exclusive).")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        self.num_qubits = num_qubits
        self.qpu = backend_qpu_type.lower()
        self.threshold = threshold
        self.verbose = verbose
        self._sim_backend = None
        self.solver = None
        self.sim_backend = sim_backend.lower()
        modeller = BuildModel(
            noise_model = noise_model,
            num_qubits = self.num_qubits,
            num_cores = int(os.cpu_count() // 2),
            threshold = self.threshold,
            basis_gates = QuantumCircuit.basis_gates_set[self.qpu]["basis_gates"],
            verbose = self.verbose
        )
        single_error, multi_error, measurement_error, connectivity = modeller.build_qubit_gate_model()
        self.single_qubit_error = {
                q : {gate : payload["qubit_channel"] for gate, payload in gates.items()} for q, gates in single_error.items()
            }
        if self.sim_backend in ["pennylane"]:
            self.two_qubit_error = {
                gate : {pair : convert_matrix_to_little_endian(payload["qubit_channel"]) for pair, payload in pairs.items()} for gate, pairs in multi_error.items()
            }
        else:
            self.two_qubit_error = {
                gate : {pair : payload["qubit_channel"] for pair, payload in pairs.items()} for gate, pairs in multi_error.items()
            }
        self.measurement_error = measurement_error
        self.measurement_error_operator = None
        self.connectivity = connectivity
        self._gate_decomposer = QuantumCircuit.basis_gates_set[self.qpu]["gate_decomposition"](
            num_qubits = self.num_qubits,
            connectivity = self.connectivity,
            qubit_map = modeller.qubit_coupling_map
        )
        self._ray_initialized = False

    @property
    def sim_backend(self)->str:
        """
        Getter for the _sim_backend attribute

        Returns
        --------
        str
            The name of the current simulation backend.
        """
        return self._sim_backend
    
    @sim_backend.setter
    def sim_backend(self,
                    backend:str)->None:
        """
        Setter for the _sim_backend attribute and updates the solver modules.

        Parameters
        -----------
        backend : str
            The name of the simulation backend to use. Supported options are "custom", "pennylane", "qiskit" and "qulacs".
        
        Raises
        -------
        TypeError 
            Raised when backend is not a string
        ValueError
            Raised when the specified backend is not available.
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

    def __getattr__(self, 
                    name: str
                    ) -> Callable:
        """
        Delegate unknown attributes/methods to the selected methods class.

        Returns
        -------
        Callable
            The method corresponding to the gate name if it exists in the gate decomposer, otherwise raises an AttributeError.
        """
        if name is not None:
            return getattr(self._gate_decomposer, name)

    def refresh(self):
        """
        Resets the quantum circuit by clearing the instruction list and qubit-to-instruction mapping.
        """
        self._gate_decomposer.instruction_list = []
    
    def _initialize_ray(self,
                        num_cores:int
                        )->None:
        """
        Initializes the Ray parallel execution environment.

        Parameters
        -----------
        num_cores : int
            The number of CPU cores to use for parallel execution.
        """
        ray.init(num_cpus=num_cores, ignore_reinit_error=True, log_to_driver=False)
        single_qubit_noise_array_ref = ray.put(np.array(list(self.single_qubit_error.items())))
        two_qubit_noise_array_ref = ray.put(np.array(list(self.two_qubit_error.items())))
        self.workers = [
            self.solver.RemoteExecutor.remote(
                num_qubits = self.num_qubits,
                single_qubit_noise = single_qubit_noise_array_ref,
                two_qubit_noise = two_qubit_noise_array_ref
            ) for _ in range(num_cores)
        ]
        self._ray_initialized = True

    def execute(self,
                qubits:list[int]=None,
                num_trajectories:int=100,
                num_cores:int=-1
                )->np.ndarray[np.float64]:
        """
        Executes the quantum circuit simulation using the Monte-Carlo Wavefunction method.

        Parameters
        -----------
        qubits : list[int], optional
            List of qubits to be measured. If None, all qubits will be measured.
        num_trajectories : int, optional
            The total number of trajectories to run. Defaults to 100.
        num_cores : int, optional
            The number of CPU cores to use for parallel execution. If -1, half of all available cores will be used. Defaults to -1.
        
        Returns
        --------
        np.ndarray[np.float64]
            An array containing the probabilities of the output states after executing the quantum circuit.

        Raises
        -------
        TypeError
            - Raised when qubits is not a list of integers.
            - Raised when num_trajectories is not an integer.
            - Raised when num_cores is not an integer.
        ValueError
            - Raised when qubits contains invalid qubit indices.
            - Raised when there are no instructions in the circuit to execute.
            - Raised when num_trajectories is not a positive integer.
            - Raised when num_cores is not a positive integer or -1.
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if not isinstance(num_trajectories, int):
            raise TypeError("num_trajectories must be an integer.")
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be a positive integer.")
        if not isinstance(num_cores, int):
            raise TypeError("num_cores must be an integer.")
        if num_cores < 1 and num_cores != -1:
            raise ValueError("num_cores must be a positive integer or -1 for all available cores.")
        if num_cores == -1 or num_cores > os.cpu_count():
            print(f"Utilizing half of all available CPU cores: {os.cpu_count() // 2} cores.")
            num_cores = os.cpu_count() // 2
        if self.sim_backend == "custom":
            solver = self.solver.RemoteExecutor(
                num_qubits = self.num_qubits,
                single_qubit_noise = self.single_qubit_error,
                two_qubit_noise = self.two_qubit_error,
                num_cores = num_cores
            )
            probs = solver.run(
                num_trajectories = num_trajectories,
                instruction_list = self.instruction_list,
                qubits = qubits
            )
        else:
            if not self._ray_initialized:
                self._initialize_ray(num_cores = num_cores)
            reset_probs = [
                self.workers[i].reset.remote() for i in range(num_cores)
            ]
            futures = [
                self.workers[traj_id % num_cores].run.remote(traj_id, self.instruction_list) for traj_id in range(num_trajectories)
            ]
            prob_chunks = [
                ray.get(self.workers[i].get.remote()) for i in range(num_cores)
            ]
            probs = np.sum(prob_chunks, axis=0) / num_trajectories
            self.shutdown()
        if self.sim_backend == "pennylane":
            probs = probs.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)
        if len(qubits) < self.num_qubits:
            probs = compute_marginal_probs(probs, [q for q in range(self.num_qubits) if q not in qubits])
        measurement_error_applicator.apply_measurement_error(
            probs, 
            self.measurement_error, 
            qubits, 
            len(qubits), 
            num_cores
        )
        probs = probs.reshape([2]*len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
        return probs

    def run_with_density_matrix(self, 
                                qubits:list[int],
                                num_cores:int=1
                                )->np.ndarray[np.float64]:
        """
        Runs the quantum circuit with the density matrix solver.

        Parameters
        -----------
        qubits : list[int]
            List of qubits to be measured.
        num_cores : int
            The number of CPU cores to use for parallel execution. Defaults to 1.

        Returns
        --------
        np.ndarray[np.float64]
            Probabilities of measuring each qubit in the computational basis.

        Raises
        -------
        TypeError
            - Raised when qubits is not a list of integers.
            - Raised when num_cores is not an integer.
        ValueError
            - Raised when qubits contains invalid qubit indices.
            - Raised when there are no instructions in the circuit to execute.
            - Raised when num_cores is not a positive integer or exceeds the number of available CPU cores.
        """
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if not isinstance(num_cores, int):
            raise TypeError("num_cores must be an integer.")
        if num_cores < 1 or num_cores > os.cpu_count():
            raise ValueError("num_cores must be a positive integer between 1 and the number of available CPU cores.")
        density_matrix_solver = self.solver.DensityMatrixSolver(
            num_qubits = self.num_qubits,
            single_qubit_noise = self.single_qubit_error,
            two_qubit_noise = self.two_qubit_error,
            instruction_list = self.instruction_list,
            num_cores = num_cores
        )
        probs = density_matrix_solver.solve(qubits=qubits)
        if self.sim_backend == "pennylane":
            probs = probs.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)
        measurement_error_applicator.apply_measurement_error(
            probs, 
            self.measurement_error, 
            qubits, 
            len(qubits), 
            num_cores
        )
        probs = probs.reshape([2]*len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
        return probs

    def run_pure_state(self, 
                       qubits:list[int],
                       num_cores:int=1,
                       return_statevector:bool=False
                    )->np.ndarray[np.float64] | np.ndarray[np.complex128]:
        """
        Runs the quantum circuit with the pure state solver.

        Parameters
        -----------
        qubits : list[int]
            List of qubits to be measured.
        num_cores : int
            The number of CPU cores to use for parallel execution. Defaults to 1.
        return_statevector : bool
            Whether to return the final statevector instead of the probabilities. Defaults to False.

        Returns
        --------
        np.ndarray[np.float64] | np.ndarray[np.complex128]
            If return_statevector is False, returns the probabilities of measuring each qubit in the computational basis. If return_statevector is True, returns the final statevector of the quantum circuit.

        Raises
        -------
        TypeError
            - Raised when qubits is not a list of integers.
            - Raised when num_cores is not an integer.
            - Raised when return_statevector is not a boolean.
        ValueError
            - Raised when qubits contains invalid qubit indices.
            - Raised when there are no instructions in the circuit to execute.
            - Raised when num_cores is not a positive integer or exceeds the number of available CPU cores.

        Notes
        ------
        If return_statevector is set to True, the function will return the final statevector of the quantum circuit instead of the probabilities. In this case, the qubits argument will be ignored and the statevector for all qubits will be returned.
        """
        if not isinstance(return_statevector, bool):
            raise TypeError("return_statevector must be a boolean.")
        if not return_statevector:
            if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
                raise TypeError("qubits must be a list of integers.")
            if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
                raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if not isinstance(num_cores, int):
            raise TypeError("num_cores must be an integer.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if num_cores < 1 or num_cores > os.cpu_count():
            raise ValueError("num_cores must be a positive integer between 1 and the number of available CPU cores.")
        pure_state_solver = self.solver.PureStateSolver(
            num_qubits = self.num_qubits,
            instruction_list = self.instruction_list,
            num_cores = num_cores,
            return_statevector = return_statevector
        )
        if return_statevector:
            output = pure_state_solver.solve()
            qubits = list(range(self.num_qubits))
        else:
            output = pure_state_solver.solve()
        if len(qubits) < self.num_qubits:
            if self._sim_backend in ["pennylane"]:
                output = output.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)
            output = compute_marginal_probs(output, [q for q in range(self.num_qubits) if q not in qubits])
        if self._sim_backend in ["pennylane"] and len(qubits) == self.num_qubits:
            return output
        output = output.reshape([2]*len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
        return output
            
    
    def draw_circuit(self,
                     style:str="mpl")->None:
        """
        Draws the quantum circuit.

        Parameters
        -----------
        style : str, optional 
            The style of the drawing, either 'mpl' for matplotlib or 'text' for text-based representation. Defaults to 'mpl'.
        
        Raises
        -------
        TypeError: 
            - If style is not a string.
        ValueError: 
            - If style is not one of ['mpl', 'text'].
            - If there are no instructions in the circuit to draw.
        """
        if not isinstance(style, str):
            raise TypeError("Style must be a string.")
        if style.lower() not in ["mpl", "text"]:
            raise ValueError("Style must be one of ['mpl', 'text'].")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to draw.")
        from qiskit import QuantumCircuit as QiskitQuantumCircuit
        import matplotlib.pyplot as plt
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
            plt.show()
        else:
            print(circuit.draw())
        del circuit
        gc.collect()
    
    def shutdown(self):
        """
        Shutsdown the Ray parallel execution environment.
        """
        ray.shutdown()
        self._ray_initialized = False