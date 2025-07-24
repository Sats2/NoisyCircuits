import pennylane as qml
from pennylane import numpy as np


class PureStateSolver:
    def __init__(self,
                 num_qubits:int,
                 instruction_list:list)->None:
        self.num_qubits = num_qubits
        self.instruction_list = instruction_list
        
    def solve(self)->np.ndarray:
        dev = qml.device("lightning.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def run_circuit():
            for instruction in self.instruction_list:
                qml.apply(instruction)
            return qml.probs(wires=range(self.num_qubits))
        
        probs = run_circuit()
        return probs