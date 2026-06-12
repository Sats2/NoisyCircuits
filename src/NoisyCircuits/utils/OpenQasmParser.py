"""
Helper module to parse OpenQASM 3.0 files and extract the quantum circuit instructions. 

Example:
    >>> from NoisyCircuits.utils import Parser
    >>> parser = Parser(file_path="path/to/circuit.qasm", instruction_list=[], append_to_circuit=False, basis_gates=[["x", "sx", "rz"], ["ecr"]])
    >>> circuit_instructions = parser.parse()
"""

import openqasm3
import numpy as np
from openqasm3 import ast, parser
import os


class NotAnEmptyCircuitError(Exception):
    """
    Error raised when the Quantum Circuit already contains instructions and append_circuit is set to False.
    """
    pass

class Parser:
    """
    Helper Class to parse OpenQASM 3.0 files to extract the quantum circuit and write it to the instruction list for use in the simulator.

    Parameters
    ----------
    file_path : str
        The path to the OpenQASM 3.0 file to be parsed.
    instruction_list : list
        The list to which the parsed instructions will be appended.
    append_circuit : bool
        A flag to indicate whether to add instructions to an existing quantum circuit or whether it needs to be a new circuit.
    basis_gates : list[list[str]]
        The basis gates of the quantum device.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    TypeError
        If append_circuit is not a boolean.
    NotAnEmptyCircuitError
        If the Quantum Circuit already contains instructions and append_circuit is set to False.
    ValueError
        If basis_gates is None.
    """
    def __init__(self,
                file_path : str,
                instruction_list : list,
                append_to_circuit : bool = False,
                basis_gates : list[list[str]] = None
                )->None:
        """
        Initializes the Parser object with the given file path, instruction list, and append circuit flag.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.file_path = file_path
        if not isinstance(append_to_circuit, bool):
            raise TypeError("append_to_circuit must be a boolean.")
        if instruction_list != [] and not append_to_circuit:
            raise NotAnEmptyCircuitError("The Quantum Circuit already contains instructions. Cannot add instructions when append_circuit is set to False")
        self.instruction_list = instruction_list
        if basis_gates is None:
            raise ValueError("basis_gates cannot be None. Please provide a list of supported gates.")
        self.basis_gates = basis_gates

    def _check_instructions_against_supported_gates(self,
                                                    instruction_list:list[list[str, list[int], float|np.ndarray]]
                                                    )->None:
        """
        Helper private method to check if the parsed instructions are supported by the simulator based on the provided basis gates.

        Parameters
        ----------
        instruction_list : list[list[str, list[int], float|np.ndarray]]
            The list of parsed instructions to be checked against the supported gates.
        
        Raises
        ------
        ValueError
            If any of the parsed instructions contain gates that are not supported by the simulator.
        """
        supported_gates = [gate for sublist in self.basis_gates for gate in sublist]
        unsupported_gate_list = []
        supported_gates.append("unitary")
        for instruction in instruction_list:
            if instruction[0] not in supported_gates:
                unsupported_gate_list.append(instruction[0])
        if unsupported_gate_list != []:
            raise ValueError(f"The following gates are not supported by the simulator: {unsupported_gate_list}. Supported gates are: {supported_gates}")

    def parse(self)->list[list[str, list[int], float|np.ndarray]]:
        """
        Method to parse the OpenQASM file.

        Returns
        -------
        list[list[str, list[int], float|np.ndarray]]
            A list of instructions with each instruction represented by a list of gate, qubits and parameters. Generated to be compatible with the NoisyCircuits simulators.
        """
        with open(self.file_path, "r") as f:
            qasm_code = f.read()
        program_circuit = parser.parse(qasm_code)
        pi_map = {"pi" : np.pi}
        for statement in program_circuit.statements:
            if not isinstance(statement, ast.QuantumGate):
                continue
            gate_name = statement.name.name
            num_qubits_for_gate = len(statement.qubits)
            qubit_list = []
            if num_qubits_for_gate == 1:
                q = statement.qubits[0].indices[0][0].value
                qubit_list.append(q)
                qubit_list.append(q)
            else:
                for q in statement.qubits:
                    qubit_list.append(q.indices[0][0].value)
            if statement.arguments != []:
                op = statement.arguments[0].op.name
                try:
                    lhs = statement.arguments[0].lhs.name
                except AttributeError:
                    lhs = statement.arguments[0].lhs.value
                try:
                    rhs = statement.arguments[0].rhs.name
                except AttributeError:
                    rhs = statement.arguments[0].rhs.value
                if isinstance(lhs, str):
                    lhs = pi_map[lhs]
                else:
                    lhs = float(lhs)
                if isinstance(rhs, str):
                    rhs = pi_map[rhs]
                else:
                    rhs = float(rhs)
                if op == "/":
                    param = lhs / rhs
                elif op == "*":
                    param = lhs * rhs
                elif op == "+":
                    param = lhs + rhs
                elif op == "-":
                    param = lhs - rhs
                else:
                    raise ValueError(f"Unsupported operation {op} in parameter expression.")
            else:
                param = 0.0
            self.instruction_list.append([gate_name, qubit_list, param])
        self._check_instructions_against_supported_gates(self.instruction_list)    
        return self.instruction_list    