"""
This module allows users to create and simulate quantum circuits with noise models based on quantum machine calibration data in a distributed memory environment using MPI. It provides methods for adding gates, executing the circuit with Monte-Carlo simulations, and visualizing the circuit. It considers both single and two-qubit gate errors as well as measurement errors. The functionality is similar to the `QuantumCircuit` module.\n

Example:\n
"""

import numpy as np
import simulator_mpi
import measurement_error_applicator
from collections.abc import Callable
from NoisyCircuits.utils import BuildModel, Parser, compute_marginal_probs, convert_matrix_to_little_endian
from NoisyCircuits.utils.EagleDecomposition import EagleDecomposition
from NoisyCircuits.utils.HeronDecomposition import HeronDecomposition


class QuantumCircuitMPI:
    r"""
    This class allows a user to create a quantum circuit with error model from IBM machines (or an available CSV data file with the machine calibration data) where selected gates (both parameterized and non-parameterized) are implemented as methods. The gate decomposition uses the basis gates of the IBM Eagle (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)` and :math:`ECR`) / Heron (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)`, :math:`R_X(\theta)`, :math:`CZ` and :math:`RZZ(\theta)`) QPUs. Users can use custom gate decompositions/basis gates by utilizing the OpenQasm 3.0 data format to read-in quantum circuits.

    Currently, it is only possible to apply a limited selection of single and two-qubit gates to the circuit simulation. For a full list of supported gates (except when imported via OpenQasm), please refer to the Decomposition :func:`NoisyCircuits.utils.Decomposition` class documentation.

    Parameters
    ----------
    num_qubits : int
        The number of qubits in the quantum circuit.
    noise_model : dict
        A dictionary containing the raw noise model for the quantum circuit.
    use_fractional : bool
        A flag to determine whether to use fractional gates or not. Applicable to QPUs which allow fractional gates (Defaults to True)
    backend_qpu_type : str
        The QPU architecture to use. Supported options are eagle and heron. (Defaults to heron)
    basis_gates : list[list[str]]
        A list of list of basis gates given in the format [[single qubit gates], [two qubit gates]]. (Defaults to the basis gates of the Heron QPU)
    num_nodes : int
        The total number of nodes in the HPC that can be used for the computation. (Defaults to 1)
    cores_per_node : int
        The total number of cores available for computation within a single node. (Defaults to 1)
    cores_per_trajectory : int
        The total number of cores that can be used by a single trajectory at any given time. (Defaults to 1)
    threshold : float
        The threshold for pruning errors in the noise model. (Defaults to 1e-6)
    verbose : bool
        Whether to print verbose output during the noise model construction. (Defaults to False)

    Raises
    ------
    TypeError : 
        - num_qubits must be an integer.
        - noise_model must be a dictionary.
        - use_fractional must be a boolean.
        - backend_qpu_type must be a string.
        - basis_gates must be a list of list of strings.
        - num_nodes must be an integer.
        - num_nodes must be an integer.
        - cores_per_node must be an integer.
        - cores_per_trajectory must be an integer.
        - threshold must be a float.
        - verbose must be a boolean.
    ValueError :
        - num_qubits must be a positive integer.
        - backend_qpu_type must be one of the supported QPU architectures.
        - num_nodes must be :math:`\geq 1`
        - cores_per_node must be :math:`\geq 1`
        - cores_per_trajectory must be :math:`\geq 1`
        - threshold must be between 0 and 1 (exclusive).

    Notes
    -----
    - Using the default values of the arguements `num_nodes`, `core_per_node` and `cores_per_trajectory` results in a fully serial execution.
    - Users need to be aware of the MPI environment of the HPC they have access to and build necessary dependencies themselves as a pre-compiled pip package install or conda install may result in dependency clashes. (See HPC Installation guidelines for more information)
    """
    basis_gate_set = {
        "eagle" : {
            "basis_gates" : [["x", "sx", "rz"], ["ecr"]],
            "gate_decomposition" : EagleDecomposition
        },
        "heron" : {
            "basis_gates" : [["x", "sx", "rz", "rx"], ["rzz", "cz"]],
            "gate_decomposition" : HeronDecomposition
        }
    }

    def __init__(
                self,
                num_qubits : int,
                noise_model : dict,
                use_fractional : bool = True,
                backend_qpu_type : str = "heron",
                basis_gates : list[list[str]] = [["x", "sx", "rz", "rx"], ["rzz", "cz"]],
                num_nodes : int = 1,
                cores_per_node : int = 1,
                cores_per_trajectory : int = 1,
                threshold : float = 1e-6,
                verbose : bool = False
            ) -> None:
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be of type int")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be greater than or equal to 1")
        if not isinstance(noise_model, dict):
            raise TypeError("noise_model must be of type dict")
        if not isinstance(backend_qpu_type, str):
            raise TypeError("backend_qpu_type must be of type string")
        if backend_qpu_type.lower() not in list(QuantumCircuitMPI.basis_gate_set.keys()):
            raise ValueError("backend_qpu_type must in {}".format(list(QuantumCircuitMPI.basis_gate_set.keys())))
        if not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be of type int")
        if num_nodes < 1:
            raise ValueError("num_nodes must be greater than or equal to 1")
        if not isinstance(cores_per_node, int):
            raise TypeError("cores_per_node must be of type int")
        if cores_per_node < 1:
            raise ValueError("cores_per_node must be greater than or equal to 1")
        if not isinstance(cores_per_trajectory, int):
            raise TypeError("cores_per_trajectory must be of type int")
        if cores_per_trajectory < 1:
            raise ValueError("cores_per_trajectory must be greater than or equal to 1")
        if not isinstance(threshold, float):
            raise ValueError("threshold must be of type float")
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between [0, 1]")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be of type boolean")
        if cores_per_trajectory > cores_per_node:
            raise ValueError("cores_per_trajectory must be lesser than or equal to cores_per_node")
        if not isinstance(basis_gates, list) or any(not isinstance(gate_set, list) for gate_set in basis_gates):
            raise TypeError("basis_gates must be a list of list of strings. For example, the default value for basis_gates is defined as {}".format(QuantumCircuitMPI.basis_gates_set["heron"]["basis_gates"]))
        if not any(any(isinstance(gate, str) for gate in gate_set) for gate_set in basis_gates):
            raise TypeError("basis_gates must be a list of list of strings. For example, the default value for basis_gates is defined as {}".format(QuantumCircuitMPI.basis_gates_set["heron"]["basis_gates"]))
        self.num_qubits = num_qubits
        self.num_nodes = num_nodes
        self.cores_per_node = cores_per_node
        self.cores_per_trajectory = cores_per_trajectory
        self._import_mpi()
        self.verbose = verbose
        self._basis_gates = QuantumCircuitMPI.basis_gate_set[backend_qpu_type.lower()]["basis_gates"]
        self.basis_gates = self._basis_gates
        modeller = BuildModel(
            noise_model = noise_model,
            num_qubits = self.num_qubits,
            num_cores = self.cores_per_node,
            threshold = threshold,
            basis_gates = self.basis_gates,
            verbose = self.verbose
        )
        single_error, multi_error, measurement_error, connectivity = modeller.build_qubit_gate_model()
        self.single_qubit_error = {
            q : {gate : payload["qubit_channel"] for gate, payload in gates.items()} for q, gates in single_error.items()
        }
        self.two_qubit_error = {
            gate : {pair : convert_matrix_to_little_endian(payload["qubit_channel"]) for pair, payload in pairs.items()} for gate, pairs in multi_error.items()
        }
        self.measurement_error = measurement_error
        self.connectivity = connectivity
        if self.basis_gates == QuantumCircuitMPI.basis_gates_set[self.qpu]["basis_gates"]:
            self._gate_decomposer = QuantumCircuitMPI.basis_gates_set[self.qpu]["gate_decomposition"](
                num_qubits = self.num_qubits,
                connectivity = self.connectivity,
                qubit_map = modeller.qubit_coupling_map,
                use_fractional = use_fractional
            )
        else:
            print("Warning: A decomposition for the given basis gates does not exist. In-built circuit building methods are not available. Please use the OpenQasm Parser to generate the circuit.")        

    def __getattr__(
            self,
            name : str
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
    
    def refresh(self) -> None:
        """
        Resets the quantum circuit by clearing the instruction list and qubit-to-instruction mapping.
        """
        self._gate_decomposer.instruction_list = []

    def _import_mpi(self) -> None:
        """
        Imports the MPI wrapper for python.

        Raises
        ------
        ImportError : mpi4py (the MPI wrapper for Python) is not found in the environment.
        """
        global MPI
        try:
            from mpi4py import MPI
        except ImportError as e:
            raise ImportError("mpi4py is required for MPI execution but is not installed. Please install mpi4py.")
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def _check_gates_in_noise_model(self) -> None:
        """
        Private helper method to check if the gates used in the quantum circuit are supported by the noise model. Raises a ValueError if any unsupported gates are found.

        Raises
        ------
        ValueError
            If any gates used in the quantum circuit are not supported by the noise model.k
        """
        single_qubit_gates_in_model = self.single_qubit_error[0].keys()
        two_qubit_gates_in_model = self.two_qubit_error.keys()
        gate_list = [gate for sublist in [single_qubit_gates_in_model, two_qubit_gates_in_model] for gate in sublist]
        unsupported_gates = []
        for instruction in self.instruction_list:
            if instruction[0] not in gate_list and instruction[0] != "unitary":
                unsupported_gates.append(instruction[0])
        if unsupported_gates != []:
            raise ValueError(f"The following gates are supported by the noise model : {unsupported_gates}. \nSupported gates are {gate_list}")

    def read_openqasm(
            self,
            file_path : str,
            append_to_circuit : bool = False
        ) -> None:
        """
        Reads an OpenQASM 3.0 file and adds the required instructions to build the quantum circuit.

        Parameters
        ----------
        file_path : str
            The path to the OpenQASM 3.0 file.
        append_to_circuit : bool, optional
            A flag to determine whether to add the parser
        """
        try:
            parser = Parser(
                    file_path = file_path,
                    instruction_list = self._gate_decomposer.instruction_list,
                    append_to_circuit = append_to_circuit,
                    basis_gates = self.basis_gates
                )
            self._gate_decomposer.instruction_list = parser.parse()
        except:
            self.instruction_list = []
            parser = Parser(
                file_path = file_path,
                instruction_list = self.instruction_list,
                append_to_circuit = append_to_circuit,
                basis_gates = self.basis_gates
            )
            self.instruction_list = parser.parse()   
        self._check_gates_in_noise_model()     

    def execute(
            self,
            qubits : list[int] = None,
            num_trajectories : int = 100
        ) -> np.ndarray[np.float64]:
        r"""
        Executes the quantum circuit simulation using the Monte-Carlo Wavefunction method on a HPC cluster in a distributed memory environment.

        Parameters
        ----------
        qubits : list[int], optional
            List of qubits to be measured. If None, all qubits will be measured.
        num_trajectories : int
            The total number of trajectories to run. Defaults to 100.

        Returns
        -------
        np.ndarray[np.float64]
            An array containing the probabilities of the executed quantum circuit.

        Raises
        ------
        TypeError
            - Raised when qubits is not a list of integers.
            - Raised when num_trajectories is not an integer.
        ValueError
            - Raised when qubits contains invalid qubit indices.
            - Raised when there are no instructions in the circuit to execute.
            - Raised when num_trajectories is not a positive integer.
        RuntimeError
            - Raised when the total estimated MPI ranks does not equal the total number of available MPI ranks (See Notes for more information)

        Notes
        -----
        The module first estimates the total number of ranks required to safely run the simulation based on the user input data of `num_nodes` (mathematically denoted as :math:`n_N`), `cores_per_node` (mathematically denoted as :math:`c_N`) and `cores_per_trajectory` (mathematically denoted as :math:`c_T`). The total number of ranks (mathematically denoted as :math:`n_R`) required is estimated by using the following formula:

        .. math::

            n_R = n_N \cdot \lfloor\frac{c_N}{c_T}\rfloor

        The above estimate provides the minimum number of MPI ranks to safely perform the simulation. If this estimate does not match the available number of MPI ranks set by the MPI execution environment (e.g., SLURM/mpiexec), then a RuntimeError is raised in order to avoid any potential issues. 
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers")
        if any((qubit < 0 or  qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more of the qubits are out of range. The valid range is from 0 to {self.num_qubits}")
        if not isinstance(num_trajectories, int):
            raise TypeError("num_trajectories must be of type int")
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be a positive integer")
        expected_ranks = self.num_nodes * (self.cores_per_node // self.cores_per_trajectory)
        if self.size != expected_ranks:
            raise RuntimeError(
                f"QuantumCircuitMPI expects {expected_ranks} MPI ranks "
                f"(num_nodes={self.num_nodes} x cores_per_node={self.cores_per_node} // "
                f"cores_per_trajectory={self.cores_per_trajectory}) but was launched with "
                f"{self.size}. Launch with: srun -N {self.num_nodes} "
                f"--ntasks-per-node={self.cores_per_node // self.cores_per_trajectory} "
                f"--cpus-per-task={self.cores_per_trajectory} python3 <script>.py"
            )
        self._base_seed = 42
        instruction_list = self.comm.bcast(self.instruction_list if self.ranl == 0 else None, root=0)
        if not instruction_list:
            raise ValueError("No instructions in the circuit to execute")
        base = num_trajectories // self.size
        remainder = num_trajectories % self.size
        local_count = base + (1 if self.rank < remainder else 0)
        local_start = self.rank * base + min(self.rank, remainder)

        dim = 1 << self.num_qubits
        local_sum = np.zeros(dim, np.float64)
        trajectory_buffer = np.zeros(dim, dtype=np.complex128)
        if self.verbose:
            print(f"Starting Simulations: {num_trajectories} trajectories across {self.size} ranks", flush=True)
        for trajectory in range(local_count):
            trajectory_buffer.fill(0)
            simulator_mpi.run_trajectory(
                instruction_list,
                trajectory_buffer,
                self.single_qubit_error,
                self.two_qubit_error,
                self.num_qubits,
                self._base_seed + local_start + trajectory,
                self.cores_per_trajectory
            )
            local_sum += np.real(trajectory_buffer)
        total = np.zeros(dim, dtype=np.float64) if self.rank == 0 else None
        self.comm.Reduce(local_sum, total, op=MPI.SUM, root=0)
        if self.verbose:
            print("Completed all trajectories", flush=True)
        if self.rank == 0:
            total = (total / num_trajectories).astype(np.float64, order = "C")
            if len(qubits) < self.num_qubits:
                total = compute_marginal_probs(total, [q for q in range(self.num_qubits) if q not in qubits])
            measurement_error_applicator.apply_measurement_error(
                total,
                self.measurement_error,
                qubits,
                len(qubits),
                self.cores_per_node
            )
            total = total.reshape([2] * len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
        total = self.comm.bcast(total, root = 0)
        return total