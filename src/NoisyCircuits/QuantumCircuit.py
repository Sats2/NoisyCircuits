import pennylane as qml
from pennylane.devices.default_qubit import DefaultQubit
from pennylane import numpy as np
from utils.BuildQubitGateModel import BuildModel


class QuantumCircuit:
    """
    This class allows a user to create a quantum circuit with error model from IBM machines with the Eagle R3 chipset using Pennylane where selected
    gates (both parameterized and non-parameterized) are implemented as methods. The gate decomposition uses RZ, SX and X gates for single qubit operations
    and ECR gate for two qubit operations as the basis gates.
    """
    def __init__(self,
                 num_qubits:int,
                 noise_model:dict,
                 num_shots:int,
                 kraus_implementation:str="Stinespring")->None:
        """
        Initializes the QuantumCircuit with the number of qubits, noise model, number of shots, and kraus implementation.

        The allowed values for the kraus implementation are "Stinespring" and "QMWF".
        "Stinespring" uses the Stinespring Dilation theorem to represent the kraus operators.
        "QMWF" uses the Quantum Monte-Carlo Wavefunction method to represent the kraus operators.
        Note that the Stinespring implementation becomes an expensive operation for simulating a large number of qubits and is not recommended for large circuits.

        Args:
            num_qubits (int): The number of qubits in the circuit.
            noise_model (dict): The noise model to be used for the circuit.
            num_shots (int): Number of shots for the circuit execution.
            kraus_implementation (str, optional): The Kraus implementation to be used. Defaults to "Stinespring".

        Raises:
            TypeError: Raised if the number of qubits is not an integer
            TypeError: Raised if the noise model is not a dictionary
            TypeError: Raised if the number of shots is not an integer
            TypeError: Raised if the kraus implementation is not a string
            ValueError: Raised if the number of qubits is not a positive integer
            ValueError: Raised if the number of shots is not a positive integer
            ValueError: Raised if the kraus implementation is not one of the allowed values.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if not isinstance(noise_model, dict):
            raise TypeError("noise_model must be a dictionary")
        if not isinstance(num_shots, int):
            raise TypeError("num_shots must be an integer")
        if not isinstance(kraus_implementation, str):
            raise TypeError("kraus_implementation must be a string")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer")
        if num_shots <= 0:
            raise ValueError("num_shots must be a positive integer")
        if kraus_implementation not in ["Stinespring", "QMWF"]:
            raise ValueError("kraus_implementation must be either 'Stinespring' or 'QMWF'")
        self.num_qubits = num_qubits
        self.num_shots = num_shots
        if kraus_implementation == "Stinespring":
            self.kraus_implementation = self._stinespring_dilation_implementation
        else:
            self.kraus_implementation = self._quantum_monte_carlo_implementation
        self.gate_model = BuildModel(noise_model).build_qubit_gate_model()
        self.tape_instructions = []
        dev = DefaultQubit(wires=self.num_qubits)
        self.executor, _ = dev.preprocess()
        self.ancilla_idx = self.num_qubits + 1

    
    def _qmwf_step(self,
                   state:np.ndarray,
                   kraus_ops:list,
                   probs:list)->np.ndarray:
        """
        Performs a single step of the Quantum Monte-Carlo Wavefunction method.

        Args:
            state (np.ndarray): Current state of the quantum circuit.
            kraus_ops (list): List of Kraus Operators.
            probs (list): List of probabilities for each Kraus operator.

        Returns:
            np.ndarray: Updated state after applying the selected Kraus operator.
        """
        choice = np.random.choice(range(len(kraus_ops)), p=probs)
        psi = np.dot(kraus_ops[choice], state)
        return psi / np.linalg.norm(psi)
    
    def _get_kraus_probs(self,
                         kraus_ops:list,
                         state:np.ndarray)->list:
        """
        Computes the probabilities of each Kraus operator given the current state.

        Args:
            kraus_ops (list): List of Kraus Operators.
            state (np.ndarray): Current state of the quantum circuit.

        Returns:
            list: Probabilities of each Kraus operator.
        """
        probs = [np.dot(np.dot(op.conj().T, state.conj()), np.dot(op, state)) for op in kraus_ops]
        return probs

    def _quantum_monte_carlo_implementation(self, 
                                            state:np.ndarray,
                                            kraus_ops:list,
                                            N:int=1000)->np.ndarray:
        """
        Implements the Quantum Monte-Carlo Wavefunction method for simulating the Kraus Noise for the given gate.

        Args:
            state (np.ndarray): Current State of the quantum circuit.
            kraus_ops (list): List of Kraus Operators.
            N (int, optional): Number of Monte-Carlo Steps to perform. Defaults to 1000.

        Returns:
            np.ndarray: Updated state after applying the selected Kraus operator.
        """
        psi_out = np.zeros_like(state, dtype=complex)
        probs = self._get_kraus_probs(kraus_ops, state)
        for _ in range(N):
            psi_out += self._qmwf_step(state, kraus_ops, probs)
        psi_out = psi_out / N
        return psi_out / np.linalg.norm(psi_out)
    
    def _stinespring_dilation_implementation(self,
                                             state:np.ndarray,
                                             kraus_ops:list)->np.ndarray:
        raise NotImplementedError()
    
    def _prepare_ancilla(self,
                         rot_angle:int|float):
        """
        Prepares the ancilla qubit for probabilistic noise application.

        Args:
            rot_angle (int | float): Angle of rotation for the ancilla qubit corresponding to the probability of the noise instruction.
        """
        self.tape_instructions.append(qml.RY(phi=rot_angle, wires=self.ancilla_idx))

    def _apply_probabilistic_noise(self,
                                   instruction_name:list,
                                   qubit:int):
        """
        Applies the required probabilistic noise instruction to the ancilla qubit and the target qubit.

        Args:
            instruction_name (list): List of noise instructions to apply.
            qubit (int): Target qubit to which the noise is applied.
        """
        for name in instruction_name:
            self.tape_instructions.append(name(wires=[self.ancilla_idx, qubit]))
        
    def _reset_ancilla(self,
                       rot_angle:int|float):
        """
        Resets the ancilla qubit to its initial state after applying the noise instructions.

        Args:
            rot_angle (int | float): Angle of rotation for the ancilla qubit.
        """
        self.tape_instructions.append(qml.measure(self.ancilla_idx, reset=True))

    def RZ(self,
           theta:int|float,
           qubit:int):
        """
        Implements the RZ gate.

        Args:
            theta (int | float): The angle of rotation.
            qubit (int): The target qubit.
        """
        self.tape_instructions.append(qml.RZ(phi=theta, wires=qubit))
    
    def SX(self,
           qubit:int):
        """
        Implements the SX gate.

        Args:
            qubit (int): The target qubit.
        """
        self.tape_instructions.append(qml.SX(wires=qubit))
    
    def X(self,
          qubit:int):
        """
        Implements the X gate.

        Args:
            qubit (int): The target qubit.
        """
        self.tape_instructions.append(qml.X(wires=qubit))

    def ECR(self,
            control:int,
            target:int):
        """
        Implements the ECR (Echoed Cross Resonance) gate

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.tape_instructions.append(qml.ECR(wires=[control, target]))
    
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
        Implements the CZ (Controlled-Z) gate.

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.RZ(theta=-np.pi/2, qubit=control)
        self.SX(qubit=target)
        self.RZ(theta=np.pi/2, qubit=target)
        self.ECR(control=control, target=target)
        self.X(qubit=control)
        self.RZ(theta=np.pi/2, qubit=target)
        self.SX(qubit=target)
        self.RZ(theta=np.pi/2, qubit=target)
        self.RX(theta=-np.pi/2, qubit=control)
        self.SX(qubit=control)

    def CY(self,
           control:int,
           target:int):
        """
        Implements the CY (Controlled-Y) gate.

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.RZ(theta=-np.pi/2, qubit=control)
        self.RZ(theta=np.pi/2, qubit=target)
        self.SX(qubit=target)
        self.RZ(theta=-np.pi, qubit=target)
        self.ECR(control=control, target=target)
        self.X(qubit=control)
        self.RZ(theta=np.pi/2, qubit=target)
        self.RZ(theta=np.pi, qubit=control)
        self.SX(qubit=control)
        self.RZ(theta=np.pi, qubit=control)
        self.SX(qubit=control)

    def CX(self,
           control:int,
           target:int):
        """
        Implements the CX (Controlled-X) gate.

        Args:
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.RZ(theta=-np.pi/2, qubit=control)
        self.RZ(theta=-np.pi, qubit=target)
        self.SX(qubit=target)
        self.RZ(theta=-np.pi, qubit=target)
        self.ECR(control=control, target=target)
        self.X(qubit=control)
        self.RZ(theta=-np.pi, qubit=control)
        self.SX(qubit=control)
        self.RZ(theta=np.pi, qubit=control)
        self.SX(qubit=control)

    def SWAP(self,
             qubit1:int,
             qubit2:int):
        """
        Implements the SWAP gate.

        Args:
            qubit1 (int): The first qubit.
            qubit2 (int): The second qubit.
        """
        self.RZ(theta=-np.pi/2, qubit=qubit1)
        self.SX(qubit=qubit2)
        self.ECR(control=qubit1, target=qubit2)
        self.SX(qubit=qubit1)
        self.RZ(theta=-np.pi/2, qubit=qubit2)
        self.ECR(control=qubit2, target=qubit1)
        self.RZ(theta=-np.pi/2, qubit=qubit1)
        self.SX(qubit=qubit1)
        self.ECR(control=qubit1, target=qubit2)

    def CRX(self,
            theta:int|float,
            control:int,
            target:int):
        """
        Implements the CRX (Controlled-RX) gate.

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.RZ(theta=-np.pi/2, qubit=control)
        self.RZ(theta=-np.pi/2, qubit=target)
        self.SX(qubit=target)
        self.RZ(theta=(np.pi - theta)/2, qubit=target)
        self.SX(qubit=target)
        self.ECR(control=control, target=target)
        self.X(qubit=control)
        self.RZ(theta=np.pi - theta/2, qubit=target)
        self.RZ(theta=-np.pi/2, qubit=control)
        self.SX(qubit=target)
        self.RZ(theta=-np.pi, qubit=target)
        self.ECR(control=control, target=target)
        self.X(qubit=control)
        self.RZ(np.pi/2, qubit=target)
        self.SX(qubit=target)
        self.RZ(np.pi/2, qubit=target)
        self.SX(qubit=control)
        self.RZ(theta=np.pi, qubit=control)
        self.SX(qubit=control)
        self.RZ(np.pi, qubit=control)

    def CRY(self,
            theta:int|float,
            control:int,
            target:int):
        """
        Implements the CRY (Controlled-RY) gate.

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.RZ(theta=-np.pi/2, qubit=control)
        self.RZ(theta=-np.pi, qubit=target)
        self.SX(qubit=target)
        self.RZ(theta=np.pi - theta/2, qubit=target)
        self.ECR(control=control, target=target)
        self.X(qubit=control)
        self.RZ(theta=-np.pi, qubit=target)
        self.RZ(theta=-np.pi/2, qubit=control)
        self.SX(qubit=target)
        self.RZ(theta=-np.pi + theta/2, qubit=target)
        self.ECR(control=control, target=target)

    def CRZ(self,
            theta:int|float,
            control:int,
            target:int):
        """
        Implements the CRZ (Controlled-RZ) gate.

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        """
        self.RZ(theta=-np.pi/2, qubit=control)
        self.RZ(theta=-np.pi + theta/2, qubit=target)
        self.SX(qubit=target)
        self.RZ(theta=-np.pi, qubit=target)
        self.ECR(control=control, target=target)
        self.X(qubit=control)
        self.RZ(theta=np.pi - theta/2, qubit=target)
        self.SX(qubit=target)
        self.RZ(theta=-np.pi, qubit=target)
        self.ECR(control=control, target=target)
        self.RZ(theta=-np.pi/2, qubit=control)

    def get_state(self)-> np.ndarray:
        """
        Returns the current state of the quantum circuit. And resets the circuit.

        Returns:
            np.ndarray: The current state of the quantum circuit.
        """
        return qml.state()

    def get_probs(self)->np.ndarray:
        """
        Returns the probabilities of the current state of the quantum circuit.

        Returns:
            np.ndarray: The probabilities of the current state of the quantum circuit.
        """
        return qml.probs(wires=range(self.num_qubits))