import numpy as np
from NoisyCircuits.utils.SwapSequence import QubitCouplingMap
from NoisyCircuits.utils.Decomposition import Decomposition

class EagleDecomposition(Decomposition):
    """
    Class for Quantum Gate Application for the IBM Eagle QPU Architectures
    """
    def __init__(self,
                 num_qubits:int,
                 connectivity:dict,
                 qubit_map:list[tuple]):
        """
        Constructor for the EagleDecomposition class that applies the quantum gates for the IBM Eagle QPU Architectures.

        Args:
            num_qubits (int): The number of qubits in the quantum circuit.
            connectivity (dict): A dictionary representing the connectivity of the qubits.
            qubit_map (list[tuple]): A list of tuples representing the mapping of logical qubits to physical qubits.
        """
        super().__init__(num_qubits=num_qubits)
        self.instruction_list = []
        self.num_qubits = num_qubits
        self.connectivity = connectivity
        self.qubit_map = qubit_map
        self.qubit_coupling = QubitCouplingMap(num_qubits=self.num_qubits, connectivity=self.connectivity)

    def RZ(self, 
           theta:int|float,
           qubit:int)->None:
        if super().RZ(theta=theta, qubit=qubit):
            self.instruction_list.append(["rz", [qubit], theta])

    def SX(self,
           qubit:int)->None:
        if super().SX(qubit=qubit):
            self.instruction_list.append(["sx", [qubit], None])

    def X(self,
          qubit:int)->None:
        if super().X(qubit=qubit):
            self.instruction_list.append(["x", [qubit], None])

    def RY(self,
           theta:int|float,
           qubit:int)->None:
        self.X(qubit=qubit)
        self.SX(qubit=qubit)
        self.RZ(theta=-theta, qubit=qubit)
        self.SX(qubit=qubit)

    def RX(self,
           theta:int|float,
           qubit:int):
        self.RZ(theta=np.pi/2, qubit=qubit)
        self.SX(qubit=qubit)
        self.RZ(theta=2*np.pi + theta, qubit=qubit)
        self.SX(qubit=qubit)
        self.RZ(theta=5*np.pi/2, qubit=qubit)
        self.X(qubit=qubit)

    def H(self,
          qubit:int):
        self.SX(qubit=qubit)
        self.RZ(theta=np.pi/2, qubit=qubit)
        self.SX(qubit=qubit)

    def Z(self,
          qubit:int):
        self.H(qubit=qubit)
        self.RZ(theta=np.pi/2, qubit=qubit)

    def Y(self,
          qubit:int):
        self.RY(theta=np.pi, qubit=qubit)
        self.RX(theta=np.pi, qubit=qubit)
        self.X(qubit=qubit)

    def ECR(self,
            control:int,
            target:int):
        if super().ECR(control=control, target=target):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            match_qubits = next((t for t in self.qubit_map if phys_control in t and phys_target in t), None)
            if phys_control == match_qubits[0] and phys_target == match_qubits[1]:
                self.instruction_list.append(["ecr", [phys_control, phys_target], None])
            else:
                self.RZ(theta=np.pi/2, qubit=phys_control)
                self.RZ(theta=-np.pi/2, qubit=phys_target)
                self.SX(qubit=phys_control)
                self.SX(qubit=phys_target)
                self.RZ(theta=-np.pi/2, qubit=phys_control)
                self.RZ(theta=np.pi/2, qubit=phys_target)
                self.instruction_list.append(["ecr", [phys_target, phys_control], None])
                self.RZ(theta=np.pi/2, qubit=phys_control)
                self.RZ(theta=np.pi/2, qubit=phys_target)
                self.SX(qubit=phys_control)
                self.SX(qubit=phys_target)
                self.RZ(theta=np.pi/2, qubit=phys_control)
                self.RZ(theta=np.pi/2, qubit=phys_target)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def apply_swap_decomposition(self, qubit1, qubit2):
        self.RZ(theta=-np.pi/2, qubit=qubit1)
        self.SX(qubit=qubit2)
        self.ECR(control=qubit1, target=qubit2)
        self.SX(qubit=qubit1)
        self.RZ(theta=-np.pi/2, qubit=qubit2)
        self.ECR(control=qubit2, target=qubit1)
        self.RZ(theta=-np.pi/2, qubit=qubit1)
        self.SX(qubit=qubit2)
        self.ECR(control=qubit1, target=qubit2)

    def CZ(self,
           control:int,
           target:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.RX(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_control)
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def CY(self,
           control:int,
           target:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
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
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def CX(self,
           control:int,
           target:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
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
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def RZZ(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=qubit1, logical_target=qubit2)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=(theta - np.pi), qubit=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RY(theta=2*np.pi, qubit=phys_control)
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def RYY(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=qubit1, logical_target=qubit2)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.RZ(theta=-np.pi, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi-theta, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(theta=-np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RX(theta=-np.pi, qubit=phys_control)
        self.X(qubit=phys_control)
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def RXX(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=qubit1, logical_target=qubit2)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.RZ(theta=np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=3*np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=np.pi+theta, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def SWAP(self,
             qubit1:int,
             qubit2:int):
        forward_swaps, reverse_swaps, qubits_1, qubits_2 = self.qubit_coupling.generate_swap_sequence(logical_control=qubit1, logical_target=qubit2)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.apply_swap_decomposition(qubit1=qubits_1, qubit2=qubits_2)
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def CRX(self, 
            theta:int|float, 
            control:int, 
            target:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=(np.pi - theta)/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi-theta/2, qubit=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta==np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_control)
        self.RZ(theta=np.pi, qubit=phys_control)
        self.SX(qubit=phys_control)
        self.RZ(theta=np.pi, qubit=phys_control)
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def CRY(self,
            theta:int|float,
            control:int,
            target:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=np.pi - theta/2, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi + theta/2, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)

        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def CRZ(self,
            theta:int|float,
            control:int,
            target:int):
        forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
        for swap in forward_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
        self.RZ(theta=-np.pi/2, qubit=phys_control)
        self.RZ(theta=-np.pi+theta/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.X(qubit=phys_control)
        self.RZ(theta=np.pi-theta/2, qubit=phys_target)
        self.SX(qubit=phys_target)
        self.RZ(theta=-np.pi, qubit=phys_target)
        self.ECR(control=phys_control, target=phys_target)
        self.RZ(theta=-np.pi/2, qubit=phys_target)
        for swap in reverse_swaps:
            self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def apply_unitary(self, 
                      unitary_matrix:np.ndarray, 
                      qubits:list[int]):
        if super().apply_unitary(unitary_matrix=unitary_matrix, qubits=qubits):
            self.instruction_list.append(["unitary", qubits, unitary_matrix])