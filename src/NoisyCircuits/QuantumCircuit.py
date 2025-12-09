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

import pennylane as qml
from pennylane import numpy as np
from NoisyCircuits.utils.BuildQubitGateModel import BuildModel
from NoisyCircuits.utils.DensityMatrixSolver import DensityMatrixSolver
from NoisyCircuits.utils.PureStateSolver import PureStateSolver
from NoisyCircuits.utils.ParallelExecutor import RemoteExecutor
import json
import ray


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
        TypeError: If threshold is not a float.
        ValueError: If threshold is not between 0 and 1 (exclusive).
        TypeError: If num_cores is not an integer.
        ValueError: If num_cores is less than 1.
        TypeError: If jsonize is not a boolean.
        TypeError: If verbose is not a boolean.
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
        if num_cores < 1:
            raise ValueError("Number of cores must be a positive integer greater than or equal to 1.")
        if not isinstance(jsonize, bool):
            raise TypeError("Jsonize must be a boolean.")
        if not isinstance(verbose, bool):
            raise TypeError("Verbose must be a boolean.")
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        if jsonize:
            self.noise_model = json.JSONDecoder().decode(json.dumps(noise_model))
        self.num_trajectories = num_trajectories
        self.threshold = threshold
        self.num_cores = num_cores
        self.instruction_list = []
        self.qubit_to_instruction_list = []
        single_error, multi_error, measure_error, connectivity = BuildModel(
            noise_model=self.noise_model,
            num_qubits=self.num_qubits,
            threshold=self.threshold
        ).build_qubit_gate_model()
        self.single_qubit_instructions = single_error
        self.two_qubit_instructions = multi_error
        self.measurement_error = measure_error
        self.connectivity = connectivity
        self.qubit_coupling_map = modeller.qubit_coupling_map
        self.measurement_error_operator = self._generate_measurement_error_operator()
        self._gate_decomposer = QuantumCircuit.basis_gates_set[self.qpu]["gate_decomposition"](
                                                                                                num_qubits=self.num_qubits,
                                                                                                connectivity=self.connectivity,
                                                                                                qubit_map=self.qubit_coupling_map
                                                                                            )
        ray.init(num_cpus=self.num_cores, ignore_reinit_error=True, log_to_driver=False)
        self.workers = [
            RemoteExecutor.remote(
                num_qubits=self.num_qubits,
                single_qubit_noise=self.single_qubit_instructions,
                two_qubit_noise=self.two_qubit_instructions
            ) for _ in range(self.num_cores)]

    def __getattr__(self, name: str) -> callable:
        """
        Apply the SWAP gate decomposition and update qubit mapping.
        
        Args:
            qubit1 (int): First qubit in SWAP
            qubit2 (int): Second qubit in SWAP
        """
        # Apply the SWAP decomposition
        self.RZ(theta=-np.pi/2, qubit=qubit1)
        self.SX(qubit=qubit2)
        self.ECR(control=qubit1, target=qubit2)
        self.SX(qubit=qubit1)
        self.RZ(theta=-np.pi/2, qubit=qubit2)
        self.ECR(control=qubit2, target=qubit1)
        self.RZ(theta=-np.pi/2, qubit=qubit1)
        self.SX(qubit=qubit2)
        self.ECR(control=qubit1, target=qubit2)
        
        # Update the qubit mapping
        self._update_mapping_after_swap(qubit1, qubit2)

    def RZ(self,
           theta:int|float,
           qubit:int):
        """
        Implements the RZ gate.

        Args:
            theta (int | float): The angle of rotation.
            qubit (int): The target qubit.
        """
        self.instruction_list.append(["rz", [qubit], theta])

    def SX(self,
           qubit:int):
        """
        Implements the SX gate.

        Args:
            qubit (int): The target qubit.
        """
        self.instruction_list.append(["sx", [qubit], None])
    
    def X(self,
          qubit:int):
        """
        Implements the X gate.

        Args:
            qubit (int): The target qubit.
        """
        self.instruction_list.append(["x", [qubit], None])

    def ECR(self,
            control:int,
            target:int):
        """
        Implements the ECR (Echoed Cross Resonance) gate

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.instruction_list.append(["ecr", [control, target], None])

    def RY(self,
           theta:int|float,
           qubit:int):
        """
        Implements the RY gate using the decomposition into SX and RZ gates.

        Args:
            theta (int | float): The angle of rotation.
            qubit (int): The target qubit.
        """
        self.X(qubit=qubit)
        self.SX(qubit=qubit)
        self.RZ(theta=-theta, qubit=qubit)
        self.SX(qubit=qubit)

    def Y(self,
           qubit:int):
        """
        Implements the Y gate using the decomposition into SX and RZ gates.

        Args:
            qubit (int): The target qubit.
        """
        self.RY(theta=np.pi, qubit=qubit)
    
    def Z(self,
          qubit:int):
        """
        Implements the exact Pauli Z gate.

        Args:
            qubit (int): The target qubit.
        """
        self.H(qubit=qubit)
        self.RY(theta=np.pi/2, qubit=qubit)

    def RX(self,
           theta:int|float,
           qubit):
        """
        Implements the RX gate using the decomposition into RZ and SX gates.

        Args:
            theta (int | float): The angle of rotation.
            qubit (int): The target qubit.
        """
        self.RZ(theta=np.pi/2, qubit=qubit)
        self.SX(qubit=qubit)
        self.RZ(theta=2*np.pi + theta, qubit=qubit)
        self.SX(qubit=qubit)
        self.RZ(theta=5*np.pi/2, qubit=qubit)
        self.X(qubit=qubit)

    def H(self,
          qubit:int):
        """
        Implements the Hadamard gate.

        Args:
            qubit (int): The target qubit.
        """
        self.SX(qubit=qubit)
        self.RZ(theta=np.pi/2, qubit=qubit)
        self.SX(qubit=qubit)

    def CZ(self,
           control:int,
           target:int):
        """
        Implements the CZ (Controlled-Z) gate with automatic routing.

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        # Generate SWAP sequence for routing
        forward_swaps, reverse_swaps, phys_control, phys_target = self._generate_swap_sequence(control, target)
        
        # Apply forward SWAPs to bring qubits together
        for swap in forward_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])
        
        # Apply the CZ gate on physical qubits
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.RX(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_control)
        
        # Apply reverse SWAPs to restore original positions
        for swap in reverse_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])

    def CY(self,
           control:int,
           target:int):
        """
        Implements the CY (Controlled-Y) gate with automatic routing.

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        # Generate SWAP sequence for routing
        forward_swaps, reverse_swaps, phys_control, phys_target = self._generate_swap_sequence(control, target)
        
        # Apply forward SWAPs to bring qubits together
        for swap in forward_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])
        
        # Apply the CY gate on physical qubits
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.RZ(theta=np.pi, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(theta=np.pi, qubit=phys_control)
        self.SX(qubit=phys_control)
        
        # Apply reverse SWAPs to restore original positions
        for swap in reverse_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])

    def CX(self,
           control:int,
           target:int):
        """
        Implements the CX (Controlled-X) gate with automatic routing.

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        # Generate SWAP sequence for routing
        forward_swaps, reverse_swaps, phys_control, phys_target = self._generate_swap_sequence(control, target)
        
        # Apply forward SWAPs to bring qubits together
        for swap in forward_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])
        
        # Apply the CX gate on physical qubits
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=-np.pi, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(theta=np.pi, qubit=phys_control)
        self.SX(qubit=phys_control)
        
        # Apply reverse SWAPs to restore original positions
        for swap in reverse_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])

    def SWAP(self,
             qubit1:int,
             qubit2:int):
        """
        Implements the SWAP gate with automatic routing.

        Args:
            qubit1 (int): The first qubit.
            qubit2 (int): The second qubit.
        """
        # Check if qubits are directly connected
        physical1 = self.logical_to_physical[qubit1]
        physical2 = self.logical_to_physical[qubit2]
        
        if physical2 in self.connectivity.get(physical1, []) or physical1 in self.connectivity.get(physical2, []):
            # Direct SWAP possible
            self._apply_swap_decomposition(physical1, physical2)
        else:
            # Need routing for SWAP - this creates a chain of SWAPs
            path = self._find_shortest_path(physical1, physical2)
            
            # Create a series of adjacent SWAPs to move qubit1 to qubit2's position
            current_pos = physical1
            for i in range(1, len(path)):
                next_pos = path[i]
                self._apply_swap_decomposition(current_pos, next_pos)
                current_pos = next_pos

    def CRX(self,
            theta:int|float,
            control:int,
            target:int):
        """
        Implements the CRX (Controlled-RX) gate with automatic routing.

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        """
        # Generate SWAP sequence for routing
        forward_swaps, reverse_swaps, phys_control, phys_target = self._generate_swap_sequence(control, target)
        
        # Apply forward SWAPs to bring qubits together
        for swap in forward_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])
        
        # Apply the CRX gate on physical qubits
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=(np.pi - theta)/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi - theta/2, qubit=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_control)
        self.RZ(theta=np.pi, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(np.pi, qubit=phys_control)
        
        # Apply reverse SWAPs to restore original positions
        for swap in reverse_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])

    def CRY(self,
            theta:int|float,
            control:int,
            target:int):
        """
        Implements the CRY (Controlled-RY) gate with automatic routing.

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        """
        # Generate SWAP sequence for routing
        forward_swaps, reverse_swaps, phys_control, phys_target = self._generate_swap_sequence(control, target)
        
        # Apply forward SWAPs to bring qubits together
        for swap in forward_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])
        
        # Apply the CRY gate on physical qubits
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi - theta/2, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi + theta/2, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        
        # Apply reverse SWAPs to restore original positions
        for swap in reverse_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])

    def CRZ(self,
            theta:int|float,
            control:int,
            target:int):
        """
        Implements the CRZ (Controlled-RZ) gate with automatic routing.

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        """
        # Generate SWAP sequence for routing
        forward_swaps, reverse_swaps, phys_control, phys_target = self._generate_swap_sequence(control, target)
        
        # Apply forward SWAPs to bring qubits together
        for swap in forward_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])
        
        # Apply the CRZ gate on physical qubits
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi + theta/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi - theta/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        
        # Apply reverse SWAPs to restore original positions
        for swap in reverse_swaps:
            self._apply_swap_decomposition(swap[0], swap[1])
    
    def apply_unitary(self,
                      unitary_matrix:np.ndarray,
                      wires:list[int])->None:
        """
        Applies a unitary operation to the specified qubits.

        Args:
            unitary_matrix (np.ndarray): The unitary matrix to be applied.
            wires (list[int]): The qubits to which the unitary operation is applied.
        
        Raises:
            AssertionError: If the provided matrix is not unitary.
        """
        assert np.allclose(np.dot(unitary_matrix, unitary_matrix.conj().T), np.eye(unitary_matrix.shape[0])), "The provided matrix is not unitary."
        self.instruction_list.append(["unitary", wires, unitary_matrix])

    def get_qubit_mapping(self) -> dict:
        """
        Returns the current logical to physical qubit mapping.
        
        Returns:
            dict: Mapping from logical to physical qubits.
        """
        return self.logical_to_physical.copy()

    def reset_qubit_mapping(self):
        """
        Resets the qubit mapping to the initial state.
        """
        self.logical_to_physical = {i: i for i in range(self.num_qubits)}
        self.physical_to_logical = {i: i for i in range(self.num_qubits)}

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
        if self._gate_decomposer.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        if len(qubits) != self.num_qubits:
            measurement_error_operator = self._generate_measurement_error_operator(qubit_list=qubits)
        else:
            measurement_error_operator = self.measurement_error_operator
        futures = [self.workers[traj_id % self.num_cores].run.remote(traj_id, self.instruction_list, qubits) for traj_id in range(num_trajectories)]
        probs_raw = np.array(ray.get(futures))
        probs = np.mean(probs_raw, axis=0)
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
        if self._gate_decomposer.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if len(qubits) != self.num_qubits:
            measurement_error_operator = self._generate_measurement_error_operator(qubit_list=qubits)
        else:
            measurement_error_operator = self.measurement_error_operator
        density_matrix_solver = DensityMatrixSolver(
            num_qubits=self.num_qubits,
            single_qubit_noise=self.single_qubit_instructions,
            two_qubit_noise=self.two_qubit_instructions,
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
        
        Raises:
            TypeError: If qubits is not a list or the items in the list are not integers.
            ValueError: If qubits contains invalid qubit indices.
            ValueError: If there are no instructions in the circuit to execute.

        Returns:
            np.ndarray: Probabilities of the output states.
        """
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers.")
        if self._gate_decomposer.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        pure_state_solver = PureStateSolver(
            num_qubits=self.num_qubits,
            instruction_list=self.instruction_list
            )
        return pure_state_solver.solve(qubits=qubits)
    
    def shutdown(self):
        """
        Shutsdown the Ray parallel execution environment.
        """
        ray.shutdown()