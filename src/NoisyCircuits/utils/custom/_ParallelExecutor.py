"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using the custom C++ simulation backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel environment. The parallel environment is available for both shared memory as well as distributed memory systems. 

The shared memory parallel execution runs a single trajectory per core and does not a single trajectory across multiple cores. The class RemoteExecutor is used to run the simulations in parallel on a shared memory system. Unlike the case with pennylane/qiskit/qulacs solvers, this does not use Ray but instead relies on OpenMP for parallelization and is directly integrated into the C++ simulation backend. The class name is unchanged from the other backend in order to maintain a consistent interface for the QuantumCircuit module.

The distributed memory parallel execution runs a single trajectory across multiple cores but inside a single node. The class MPIExecutor is used for this purpose and relies on MPI from python-mpi4py for node level parallelization. Within each node, the simulations are run in parallel.
"""
import numpy as np
from NoisyCircuits.utils import compute_marginal_probs, convert_matrix_to_little_endian
import simulator


class RemoteExecutor:
    """
    Wrapper to the module that performs the parallel execution of the quantum circuit according to the Monte-Carlo Wavefunction method.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 num_cores:int=2)->None:
        """
        Constructor for the RemoteExecutor class.

        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            single_qubit_noise (dict): Dictionary containing the single qubit noise parameters.
            two_qubit_noise (dict): Dictionary containing the two qubit noise parameters.
            num_cores (int): Number of cores to use for parallel execution. Defaults to 2.
        """
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = {
            gate: {pair: convert_matrix_to_little_endian(payload) for pair, payload in pairs.items()} 
            for gate, pairs in two_qubit_noise.items()
        }
        self.num_cores = num_cores

    def run(self,
            num_trajectories:int,
            instruction_list:list[list[str, list[int], float]],
            qubits:list[int])->np.ndarray[np.float64]:
        """
        Method that calls the C++ simulation backend.

        Args:
            num_trajectories (int): Number of trajectories to run in parallel.
            instruction_list (list[list[str, list[int], float]]): List of instructions to build the quantum circuit.
            qubits (list[int]): List of qubits that should be measured.

        Returns:
            np.ndarray[np.float64]: Output probabilities of the noisy quantum circuit simulation.
        """
        output_array = np.zeros(1 << self.num_qubits, dtype=np.complex128)
        simulator.simulate_circuit(instruction_list, output_array, self.single_qubit_noise, self.two_qubit_noise, self.num_qubits, True, num_trajectories, False, self.num_cores)
        return output_array.astype(np.float64)
    
class MPIExecutor:
    """
    Module that performs the quantum circuit simulation in a distributed memory environment with MPI. This module runs each trajectory across multiple cores inside a single node however only one trajectory per node at a given time.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 num_nodes:int,
                 num_cores:int)->None:
        """
        Constructor for the MPIExecutor class.

        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            single_qubit_noise (dict): Dictionary containing the single qubit noise parameters.
            two_qubit_noise (dict): Dictionary containing the two qubit noise parameters.
            num_nodes (int): Number of nodes to use for distributed execution.
            num_cores (int): Number of cores per node to use for parallel execution.
        """
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = two_qubit_noise
        self.num_nodes = num_nodes
        self.num_cores = num_cores
        self._import_mpi()

    def _import_mpi(self):
        """
        Helper method to import MPI modules for python.

        Raises:
            ImportError: If mpi4py is not installed.
        """
        global MPI
        try:
            from mpi4py import MPI
            import simulator_mpi as mpi_simulator
            self.simulator = mpi_simulator
        except ImportError:
            raise ImportError("mpi4py is required for MPIExecutor but is not installed. Please install mpi4py.")
        
    def run(self,
            num_trajectories:int,
            instruction_list:list[list[str, list[int], float]],
            qubits:list[int])->np.ndarray[np.float64]:
        """
        Method that call the C++ simulation backend with MPI parallelization.

        Args:
            num_trajectories (int): Number of trajectories to run in parallel.
            instruction_list (list[list[str, list[int], float]]): List of instructions to build the quantum circuit.
            qubits (list[int]): List of qubits that should be measured.

        Returns:
            np.ndarray[np.float64]: Output probabilities of the noisy quantum circuit simulation.

        Raises:
            ValueError: If the number of MPI processes does not match the number of nodes specified.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if self.size != self.num_nodes:
            raise ValueError(f"Number of MPI processes ({self.size}) must match the number of nodes specified ({self.num_nodes}).")
        if self.rank == 0:
            trajectories_per_node = num_trajectories // self.num_nodes
            remainder_trajectories = num_trajectories % self.num_nodes
            output_array = np.zeros(1 << self.num_qubits, dtype=np.complex128)
        else:
            output_array = None
            instruction_list = None
            self.single_qubit_noise = None
            self.two_qubit_noise = None
            self.num_qubits = None
            trajectories_per_node = None
            remainder_trajectories = None
        self.comm.bcast(instruction_list, root=0)
        self.comm.bcast(self.single_qubit_noise, root=0)
        self.comm.bcast(self.two_qubit_noise, root=0)
        self.comm.bcast(self.num_qubits, root=0)
        self.comm.bcast(trajectories_per_node, root=0)
        self.comm.bcast(remainder_trajectories, root=0)
        local_output_array = np.zeros(1 << self.num_qubits, dtype=np.complex128)
        for trajectory in range(trajectories_per_node):
            seed = 42 + self.rank * trajectories_per_node + trajectory
            self.simulator.run_trajectory(instruction_list, local_output_array, self.single_qubit_noise, self.two_qubit_noise, self.num_qubits, seed, self.num_cores)
            if self.rank == 0:
                self.comm.Reduce(MPI.IN_PLACE, output_array, op=MPI.SUM, root=0)
                output_array += local_output_array
            else:
                self.comm.Reduce(local_output_array, None, op=MPI.SUM, root=0)
            local_output_array.fill(0)
        self.comm.Barrier()
        for trajectory in range(remainder_trajectories):
            if self.rank == 0:
                pass
            else:
                seed = 42 + self.rank * trajectories_per_node + remainder_trajectories - trajectory
                self.simulator.run_trajectory(instruction_list, local_output_array, self.single_qubit_noise, self.two_qubit_noise, self.num_qubits, seed, self.num_cores)
                self.comm.Reduce(local_output_array, None, op=MPI.SUM, root=0)
                local_output_array.fill(0)
        self.comm.Barrier()
        if self.rank == 0:
            output_array /= num_trajectories
            return compute_marginal_probs(output_array.astype(np.float64), [i for i in range(self.num_qubits) if i not in qubits])      