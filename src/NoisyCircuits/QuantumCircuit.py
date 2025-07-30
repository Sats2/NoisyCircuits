import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pennylane as qml
from pennylane import numpy as np
from NoisyCircuits.utils.BuildQubitGateModel import BuildModel
from NoisyCircuits.utils.DensityMatrixSolver import DensityMatrixSolver
from NoisyCircuits.utils.PureStateSolver import PureStateSolver
import json
from joblib import Parallel, delayed


class QuantumCircuit:
    """
    This class allows a user to create a quantum circuit with error model from IBM machines with the Eagle R3 chipset using Pennylane where selected
    gates (both parameterized and non-parameterized) are implemented as methods. The gate decomposition uses RZ, SX and X gates for single qubit operations
    and ECR gate for two qubit operations as the basis gates.
    """
    def __init__(self,
                 num_qubits:int,
                 noise_model:dict,
                 num_trajectories:int,
                 num_cores:int=2,
                 jsonize:bool=False,
                 threshold:float=1e-12)->None:
        """
        Initializes the QuantumCircuit with the specified number of qubits, noise model, number of trajectories for Monte-Carlo simulation, and threshold for noise application.

        Args:
            num_qubits (int): The number of qubits in the circuit.
            noise_model (dict): The noise model to be used for the circuit.
            num_trajectories (int): The number of trajectories for the Monte-Carlo simulation.
            num_cores (int, optional): The number of cores to use for parallel execution. Defaults to 2.
            jsonize (bool, optional): If True, the circuit will be serialized to JSON format. Defaults to False.
            threshold (float, optional): The threshold for noise application. Defaults to 1e-12.

        Raises:
            TypeError: If num_qubits is not an integer.
            ValueError: If num_qubits is less than 1.
            TypeError: If the noise_model is not a dictionary.
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
        self.instruction_list = []
        self.qubit_to_instruction_list = []
        single_error, multi_error, measure_error, connectivity = BuildModel(
            noise_model=self.noise_model,
            num_qubits=self.num_qubits,
            threshold=self.threshold
        ).build_qubit_gate_model()
        self.single_qubit_instructions = single_error
        self.ecr_error_instruction = multi_error
        self.measurement_error = measure_error
        self.connectivity = connectivity
        # Track logical to physical qubit mapping for routing
        self.logical_to_physical = {i: i for i in range(num_qubits)}
        self.physical_to_logical = {i: i for i in range(num_qubits)}

    def _find_shortest_path(self, start: int, end: int) -> list:
        """
        Find the shortest path between two qubits using BFS.
        
        Args:
            start (int): Starting qubit
            end (int): Target qubit
            
        Returns:
            list: Path from start to end qubit
        """
        if start == end:
            return [start]
            
        # Use connectivity map directly as it's already in adjacency list format
        # Ensure all qubits are represented in the graph
        graph = {i: [] for i in range(self.num_qubits)}
        for qubit, neighbors in self.connectivity.items():
            graph[qubit] = neighbors
        
        # BFS to find shortest path
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in graph[current]:
                if neighbor == end:
                    return path + [neighbor]
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        raise ValueError(f"No path found between qubits {start} and {end}")

    def _generate_swap_sequence(self, logical_control: int, logical_target: int) -> tuple:
        """
        Generate SWAP sequence to bring qubits close enough for interaction.
        
        Args:
            logical_control (int): Logical control qubit
            logical_target (int): Logical target qubit
            
        Returns:
            tuple: (forward_swaps, reverse_swaps, final_control_pos, final_target_pos)
        """
        # Get current physical positions
        control_pos = self.logical_to_physical[logical_control]
        target_pos = self.logical_to_physical[logical_target]
        
        # Check if they're already connected
        if target_pos in self.connectivity.get(control_pos, []) or control_pos in self.connectivity.get(target_pos, []):
            return [], [], control_pos, target_pos
        
        # Find shortest path between control and target
        path = self._find_shortest_path(control_pos, target_pos)
        
        # Generate swaps to move target qubit towards control
        forward_swaps = []
        current_target_pos = target_pos
        
        # Move target qubit step by step towards control
        for i in range(len(path) - 2, 0, -1):  # Move from target towards control
            swap_pos = path[i]
            if current_target_pos != swap_pos:
                forward_swaps.append((current_target_pos, swap_pos))
                # Update qubit mappings
                self._update_mapping_after_swap(current_target_pos, swap_pos)
                current_target_pos = swap_pos
        
        final_control_pos = self.logical_to_physical[logical_control]
        final_target_pos = self.logical_to_physical[logical_target]
        
        # Generate reverse swaps (in reverse order)
        reverse_swaps = forward_swaps[::-1]
        
        return forward_swaps, reverse_swaps, final_control_pos, final_target_pos

    def _update_mapping_after_swap(self, qubit1: int, qubit2: int):
        """
        Update logical-physical mapping after a SWAP operation.
        
        Args:
            qubit1 (int): First qubit in SWAP
            qubit2 (int): Second qubit in SWAP
        """
        # Get logical qubits at these physical positions
        logical1 = self.physical_to_logical[qubit1]
        logical2 = self.physical_to_logical[qubit2]
        
        # Swap the mappings
        self.logical_to_physical[logical1] = qubit2
        self.logical_to_physical[logical2] = qubit1
        self.physical_to_logical[qubit1] = logical2
        self.physical_to_logical[qubit2] = logical1

    def _apply_swap_decomposition(self, qubit1: int, qubit2: int):
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
        self.instruction_list.append(qml.RZ(phi=theta, wires=qubit))
        self.qubit_to_instruction_list.append(("rz", qubit))

    def SX(self,
           qubit:int):
        """
        Implements the SX gate.

        Args:
            qubit (int): The target qubit.
        """
        self.instruction_list.append(qml.SX(wires=qubit))
        self.qubit_to_instruction_list.append(("sx", qubit))
    
    def X(self,
          qubit:int):
        """
        Implements the X gate.

        Args:
            qubit (int): The target qubit.
        """
        self.instruction_list.append(qml.X(wires=qubit))
        self.qubit_to_instruction_list.append(("x", qubit))

    def ECR(self,
            control:int,
            target:int):
        """
        Implements the ECR (Echoed Cross Resonance) gate

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.instruction_list.append(qml.ECR(wires=[control, target]))
        self.qubit_to_instruction_list.append(("ecr", (control, target)))

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
        self.instruction_list = []
        self.qubit_to_instruction_list = []
    
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

    def execute(self,
                qubits:list[int],
                num_trajectories:int)->np.ndarray:
        """
        Executes the built quantum circuit with the specified noise model using the Monte-Carlo Wavefunction method.

        Args:
            np.ndarray: _description_
        """

    def execute(self,
                qubits:list[int],
                num_trajectories:int)->np.ndarray:
        """
        Executes the built quantum circuit with the specified noise model using the Monte-Carlo Wavefunction method.

        Args:
            qubits (list[int]): The list of qubits to be measured.
            num_trajectories (int): The number of trajectories for the Monte-Carlo simulation (can be modified). Defaults to None and uses the class attribute.
                                    If specified, it overrides the class attribute for this execution.
        
        Raises:
            TypeError: If qubits is not a list.
            TypeError: If num_trajectories is not an integer.
            ValueError: If num_trajectories is less than 1.
            ValueError: If qubits contains invalid qubit indices.

        Returns:
            np.ndarray: The probabilities of the output states.
        """
        dev = qml.device("lightning.qubit", wires=self.num_qubits)
        if not isinstance(qubits, list):
            raise TypeError("Qubits must be a list of integers.")
        if not all(isinstance(q, int) for q in qubits):
            raise TypeError("All qubits must be integers.")
        if not all(0 <= q < self.num_qubits for q in qubits):
            raise ValueError(f"Qubits must be in the range [0, {self.num_qubits - 1}].")
        if num_trajectories is not None:
            if not isinstance(num_trajectories, int):
                raise TypeError("Number of trajectories must be an integer.")
            if num_trajectories < 1:
                raise ValueError("Number of trajectories must be a positive integer greater than or equal to 1.")
            self.num_trajectories = num_trajectories
        print(f"Creating Measurement Error Operator for observable qubits: {qubits}")
        measurement_error_operator =self.generate_measurement_operator(qubits)
        print(f"Measurement Error Operator created.\nExecuting the circuit with {self.num_trajectories} trajectories.")

        @qml.qnode(dev)
        def apply_gate(state, gate):
            """
            Applies a quantum gate to the given state.

            Args:
                state (np.ndarray): The input state vector.
                gate (qml.operation): The quantum gate to be applied.

            Returns:
                np.ndarray: Updated state after applying the gate.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            qml.apply(gate)
            return qml.state()
        
        def apply_single_noisy_gate(state, gate, qubit, gate_id):
            """
            Applies the noise operation for the given gate using the MCWF method.

            Args:
                state (np.ndarray): Input state of the system.
                gate (qml.operation): The quantum gate to be applied.
                qubit (int): Target Qubit for the noisy gate.
                gate_id (str): Name of the gate.

            Returns:
                np.ndarray: Updated state after applying the gate.
            """
            psi = state.copy()
            state_mod = apply_gate(psi, gate)
            if np.isnan(state_mod).any():
                psi = psi + np.random.normal(0, 1e-8, size=state.shape)
                state_mod = apply_gate(psi, gate)
            kraus_probs = [np.vdot(np.dot(op, state_mod), np.dot(op,state_mod)) for op in self.single_qubit_instructions[qubit][gate_id]["kraus_operators"]]
            kraus_probs = np.abs(np.array(kraus_probs) / np.sum(kraus_probs))
            chosen_op = np.random.choice(range(len(self.single_qubit_instructions[qubit][gate_id]["kraus_operators"])), p=kraus_probs)
            state_after = np.dot(self.single_qubit_instructions[qubit][gate_id]["kraus_operators"][chosen_op], state_mod) / np.sqrt(kraus_probs[chosen_op])
            return state_after

        def apply_noisy_ecr_gate(state, gate, qubit_pair):
            """
            Applies the noise operation for the given gate using the MCWF method.

            Args:
                state (np.ndarray): Input state of the system.
                gate (qml.operation): The quantum gate to be applied.
                qubit_pair (tuple[int, int]): The pair of qubits affected by the gate.

            Returns:
                np.ndarray: Updated state after applying the gate.
            """
            psi = state.copy()
            state_after = apply_gate(psi, gate)
            qubit_pair = tuple(sorted(qubit_pair))
            if np.isnan(state_after).any():
                psi = psi + np.random.normal(0, 1e-8, size=state.shape)
                state_after = apply_gate(psi, gate)
            kraus_probs = [np.vdot(np.dot(op, state_after), np.dot(op, state_after)) for op in self.ecr_error_instruction[qubit_pair]["operators"]]
            kraus_probs = np.abs(np.array(kraus_probs) / np.sum(kraus_probs))
            chosen_op = np.random.choice(range(len(self.ecr_error_instruction[qubit_pair]["operators"])), p=kraus_probs)
            state_after = np.dot(self.ecr_error_instruction[qubit_pair]["operators"][chosen_op], state_after) / np.sqrt(kraus_probs[chosen_op])
            return state_after
        
        @qml.qnode(dev)
        def get_probs(state):
            """
            Computes the probabilities of the output states after preparing the quantum state.

            Args:
                state (np.ndarray): Input state of the system.

            Returns:
                np.ndarray: Probabilities of the output states.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            return qml.probs(wires=qubits)

        def compute_trajectory(traj_id):
            """
            Computes a single trajectory of the MCWF simulation.

            Args:
                traj_id (int): ID of the trajectory.

            Returns:
                np.ndarray: Probabilities of the output states.
            """
            np.random.seed(32+traj_id)
            init_state = np.zeros(2**self.num_qubits)
            init_state[0] = 1.0
            state = init_state.copy()
            for gate, instruction in zip(self.instruction_list, self.qubit_to_instruction_list):
                gate_instruction = instruction[0]
                qubit_added = instruction[1]
                if isinstance(qubit_added, int):
                    if gate_instruction not in self.single_qubit_instructions[qubit_added].keys():
                        state = apply_gate(state.copy(), gate)
                    else:
                        state = apply_single_noisy_gate(state.copy(), gate, qubit_added, gate_instruction)
                else:
                    state = apply_noisy_ecr_gate(state.copy(), gate, qubit_added)
            p = get_probs(state)
            return p
        
        all_probs = Parallel(n_jobs=self.num_cores)(delayed(compute_trajectory)(traj_id) for traj_id in range(self.num_trajectories))
        # all_probs = [compute_trajectory(traj_id) for traj_id in range(self.num_trajectories)]
        probs = np.zeros(2**len(qubits))
        for state in all_probs:
            probs += state
        probs /= self.num_trajectories
        if measurement_error_operator is not None:
            probs = np.dot(measurement_error_operator, probs)
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
        measurement_error_operator = self.generate_measurement_operator(qubits)
        print(f"Measurement Error Operator created. \nExecuting the circuit with density matrix solver.")
        density_matrix_solver = DensityMatrixSolver(
            num_qubits=self.num_qubits,
            single_qubit_noise=self.single_qubit_instructions,
            ecr_noise=self.ecr_error_instruction,
            measurement_noise=self.measurement_error,
            instruction_list=self.instruction_list,
            qubit_instruction_list=self.qubit_to_instruction_list
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