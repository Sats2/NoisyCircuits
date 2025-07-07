import pennylane as qml
from pennylane import numpy as np
from collections import defaultdict
import math

def _map_instruction_qubits(instrs, gate_qubits):
    """
    Maps the qubits in the instructions to the specified gate qubits.

    Args:
        instrs (list): List of instructions to map. 
        gate_qubits (dict): Mapping of qubit indices to gate qubits.

    Returns:
        list: List of mapped instructions.
    """
    mapped = []
    for inst in instrs:
        mapped_inst = inst.copy()
        mapped_inst["qubits"] = [gate_qubits[q] for q in inst["qubits"]]
        mapped.append(mapped_inst)
    return mapped

def _extract_kraus(instrs):
    """
    Extracts the Kraus operators from the instructions.

    Args:
        instrs (list): List of instructions to extract Kraus operators from.

    Returns:
        list: List of Kraus operators, or None if not found.
    """
    for inst in instrs:
        if inst["name"] == "kraus":
            return [
                np.array([[complex(*e) for e in row] for row in matrix])
                for matrix in inst["params"]
            ]
    return None

class BuildModel:
    def __init__(self,
                 noise_model:dict,
                 num_qubits:int,
                 ancillary_qubits:int,
                 additional_ancilla:int):
        self.noise_model = noise_model
        self.num_qubits = num_qubits
        self.ancilla = ancillary_qubits
        self.ancilla_2 = additional_ancilla
        self.use_qubits = list(range(num_qubits))

    def _ensure_ctpt(kraus_ops:list)->bool:
        mat = np.zeros(kraus_ops[0].shape, dtype=complex, requires_grad=False)
        for op in kraus_ops:
            mat += np.dot(op.conj().T, op)
        if not np.allclose(mat, np.eye(mat.shape[0], dtype=complex), atol=1e-3):
            raise Warning("Kraus operators do not form a complete positivity trace-preserving (CPTP) map.")
        return True
    
    def extract_single_qubit_qerrors(self,
                                    data:list,
                                    target_ops:list[str],
                                    allowed_qubits:list[int])->dict:
        """
        Extracts single-qubit quantum errors for the specified gates and qubits.

        Args:
            data (list): List of Noise model entries from Qiskit JSON.
            target_ops (list[str]): Gate operations to include.
            allowed_qubits (list[int]): Qubits to include in output.

        Returns:
            dict: Nested dictionary of format: single_qubit_instructions[q][op] = {instructions, probabilities, kraus}
        """
        allowed_qubits = set(allowed_qubits)
        result = defaultdict(lambda: defaultdict(lambda: {
            "instructions" : [],
            "probabilities": [],
            "kraus" : None
        }))

        for entry in data:
            if entry.get("type") != "qerror":
                continue
            if not any(op in target_ops for op in entry.get("operations", [])):
                continue
            operations = entry["operations"]
            probabilities = entry["probabilities"]
            instruction_list = entry["instructions"]
            gate_qubits_list = entry.get("gate_qubits", [None])
            for op in operations:
                if op not in target_ops:
                    continue
                for gate_qubits in gate_qubits_list:
                    for prob, instrs in zip(probabilities, instruction_list):
                        if gate_qubits is None:
                            mapped_instrs = instrs
                            qubits = sorted(set(q for g in instrs for q in g["qubits"]))
                        else:
                            mapped_instrs = _map_instruction_qubits(instrs, gate_qubits)
                            qubits = sorted(set(q for g in mapped_instrs for q in g["qubits"]))
                        if len(qubits) != 1 or qubits[0] not in allowed_qubits:
                            continue
                        q = qubits[0]
                        instr_names = "-".join(g["name"] for g in mapped_instrs)
                        result[q][op]["instructions"].append(instr_names)
                        result[q][op]["probabilities"].append(prob)
                        if result[q][op]["kraus"] is None:
                            kraus_matrices = _extract_kraus(mapped_instrs)
                            if kraus_matrices is not None:
                                result[q][op]["kraus"] = kraus_matrices
        return dict(result)
    
    def extract_ecr_errors(self,
                           data:list, 
                           allowed_qubits:list[int])->dict:
        """
        Extracts ECR errors from the noise model data.

        Args:
            data (list): List of Noise model entries from Qiskit JSON.
            allowed_qubits (list[int]): List of allowed qubits.

        Returns:
            dict: Dictionary of extracted ECR errors.
        """
        allowed_qubits = set(allowed_qubits)
        ecr_errors = defaultdict(list)
        for entry in data:
            if entry.get("type") != "qerror":
                continue
            if "ecr" not in entry.get("operations", []):
                continue
            probabilities = entry["probabilities"]
            instructions_list = entry["instructions"]
            gate_qubits_list = entry.get("gate_qubits", [None])
            for gate_qubits in gate_qubits_list:
                if len(gate_qubits) != 2 or not set(gate_qubits).issubset(allowed_qubits):
                    # Only include valid 2 qubit gate pairs within allowed qubits
                    continue
                qpair = tuple(sorted(gate_qubits))
                for prob, instrs in zip(probabilities, instructions_list):
                    mapped_instrs = _map_instruction_qubits(instrs, gate_qubits)
                    kraus = _extract_kraus(mapped_instrs)
                    ecr_errors[qpair].append({
                        "instructions":mapped_instrs,
                        "probability": prob,
                        "kraus": kraus
                    })
        return dict(ecr_errors)
    
    #TODO: Test this function
    def build_kraus_unitary(self, kraus_ops:list)->np.ndarray:
        """
        Builds a unitary operator from the given Kraus operators.

        Args:
            kraus_ops (list): List of Kraus operators.  

        Returns:
            np.ndarray: Unitary operator constructed from the Kraus operators.
        """
        self._ensure_ctpt(kraus_ops)
        num_dim = 2 ** math.ceil(np.log2(len(kraus_ops[0])))
        V = np.zeros((num_dim, 2), dtype=complex)
        env_basis = np.eye(num_dim)[:len(kraus_ops)]
        for k,op in enumerate(kraus_ops):
            for i in range(2):
                sys_in = np.zeros(2, dtype=complex)
                sys_in[i] = 1.0
                vec = op @ sys_in
                V[:,i] += np.kron(vec, env_basis[k])
        U, _ = np.linalg.qr(V, mode="complete")
        assert U.conj().T @ U == np.eye(U.shape[0]), "Failed to build unitary from Kraus operators."
        return U
    
    def post_process_single_qubit_errors(self,
                                         single_qubit_errors:dict)->dict:
        """
        Post-processes the single-qubit errors to create a dictionary for qubits with their respective error instructions that can be directly applied without 
        further processing.

        Args:
            single_qubit_errors (dict): The dictionary of single-qubit errors extracted from the noise model.

        Returns:
            dict: Dictionary of post-processed single-qubit errors.
        """
        reset_op = lambda gamma: np.array([
                            [np.sqrt(1-gamma), 0, 0, 0],
                            [np.sqrt(gamma), 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, np.sqrt(gamma), 0, np.sqrt(1-gamma)]
                        ], dtype=complex)
        error_channels = {
            "x" : qml.CNOT,
            "y" : qml.CY,
            "z" : qml.CZ,
            "id" : None,
        }
        single_qubit_error_processed = {}
        for qubit in single_qubit_errors.keys():
            for op in single_qubit_errors[qubit].keys():
                operator_dict = {}
                instruction_list = single_qubit_errors[qubit][op]["instructions"]
                probabilities = single_qubit_errors[qubit][op]["probabilities"]
                kraus = single_qubit_errors[qubit][op]["kraus"]
                if kraus is not None:
                    kraus_unitary = self.build_kraus_unitary(kraus)
                else:
                    kraus_unitary = None
                probability_angles = []
                for prob in probabilities:
                    angle = 2 * np.arcsin(np.sqrt(prob))
                    probability_angles.append(angle)
                error_instructions = []
                for k,instr in enumerate(instruction_list):
                    instr_string = instr.split("-")
                    if "kraus" in instr_string:
                        instr_string.remove("kraus")
                    if len(instr_string) == 1:
                        current_instruction = instr_string[0]
                        if current_instruction is not "reset":
                            error_instructions.append(qml.RY(probability_angles[k], wires=self.ancilla))
                            error_instructions.append(error_channels[current_instruction](self.ancilla, qubit))
                            error_instructions.append(qml.measure(self.ancilla, reset=True))
                        if current_instruction == "reset":
                            error_instructions.append(qml.QubitUnitary(reset_op(probabilities[k]), wires=[self.ancilla, qubit]))
                            error_instructions.append(qml.measure(self.ancilla, reset=True))
                    else:
                        error_instructions.append(qml.RY(probability_angles[k], wires=self.ancilla))
                        for entry in instr_string:
                            if entry is not "reset":
                                error_instructions.append(error_channels[entry](self.ancilla, qubit))
                            else:
                                error_instructions.append(qml.QubitUnitary(reset_op(probabilities[k]), wires=[self.ancilla, qubit]))
                        error_instructions.append(qml.measure(self.ancilla, reset=True))
                operator_dict["error_instructions"] = error_instructions
                if kraus_unitary is not None:
                    operator_dict["kraus_unitary"] = [
                                                        qml.QubitUnitary(kraus_unitary, wires=[self.ancilla, self.ancilla_2, qubit]),
                                                        qml.measure([self.ancilla, self.ancilla_2, qubit], reset=True)
                                                    ]
                else:
                    operator_dict["kraus_unitary"] = None
                single_qubit_error_processed[qubit][op] = operator_dict
        return single_qubit_error_processed

    def post_process_ecr_errors(self,
                                ecr_error:dict,
                                threshold:float=None)->dict:
        """
        Post-processes the ECR errors to create a dictionary for qubits with their respective error instructions that can be directly applied without
        further processing.

        Args:
            ecr_error (dict): The dictionary of ECR errors extracted from the noise model.
            threshold (float): Cutoff threshold for probabilities to filter out low-probability errors (applied only to ECR gates). 
                                Default is None, which means no filtering.

        Returns:
            dict: Dictionary of post-processed ECR errors.
        """
        ecr_error_processed = {}
        pauli_param_map = {
            "I" : None,
            "X" : qml.X,
            "Y" : qml.Y,
            "Z" : qml.Z
        }
        error_channels = {
            "id" : None,
            "x" : qml.CNOT,
            "y" : qml.CY,
            "z" : qml.CZ
        }
        reset_op = lambda gamma: np.array([
                            [np.sqrt(1-gamma), 0, 0, 0],
                            [np.sqrt(gamma), 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, np.sqrt(gamma), 0, np.sqrt(1-gamma)]
                        ], dtype=complex)
        for (q0, q1), error_list in ecr_error.items():
            if threshold is not None:
                filtered_errors = [e for e in error_list if e["probability"] >= threshold]
                total_probability = sum(e["probability"] for e in filtered_errors)
                for e in filtered_errors:
                    e["probability"] /= total_probability
            else:
                filtered_errors = error_list
            error_instructions = []
            kraus_unitary = []
            for instr in filtered_errors[0]["instructions"]:
                if instr["name"] == "kraus":
                    qubit = instr["qubits"][0]
                    kraus_ops = _extract_kraus(instr["params"])
                    unitary_op = self.build_kraus_unitary(kraus_ops)
                    kraus_unitary.append(
                        qml.QubitUnitary(unitary_op, wires=[self.ancilla, self.ancilla_2, qubit]),
                        qml.measure([self.ancilla, self.ancilla_2], reset=True)
                    )
            for error in filtered_errors:
                for instr in error["instructions"]:
                    rotation_angle = 2 * np.arcsin(np.sqrt(error["probability"]))
                    if instr["name"] == "pauli":
                        param_gate_1, param_gate_2 = instr["params"]
                        qubit_0, qubit_1 = instr["qubits"]
                        error_instructions.append(
                            qml.RY(rotation_angle, wires=self.ancilla),
                            pauli_param_map[param_gate_1](qubit_0),
                            qml.measure(self.ancilla, reset=True),
                            qml.RY(rotation_angle, wires=self.ancilla),
                            pauli_param_map[param_gate_2](qubit_1),
                            qml.measure(self.ancilla, reset=True)
                        )
                    else:
                        qubit = instr["qubits"][0]
                        if instr["name"] == "reset":
                            error_instructions.append(qml.QubitUnitary(reset_op(error["probability"]), wires=[self.ancilla, qubit]),
                                                      qml.measure(self.ancilla, reset=True))
                        elif instr["name"] not in ["kraus", "reset"]:
                            error_instructions.append(qml.RY(rotation_angle, wires=self.ancilla),
                                                      error_channels[instr["name"]](self.ancilla, qubit),
                                                      qml.measure(self.ancilla, reset=True))
                        else:
                            continue
                ecr_error_processed[(q0, q1)] = {
                    "error_instructions": error_instructions,
                    "kraus_unitary": kraus_unitary
                }
        return ecr_error_processed
                                    

    def build_qubit_gate_model(self, 
                               threshold:float=None)->tuple[dict, dict]:
        """
        Builds the qubit gate model by extracting single-qubit and ECR errors from the noise model.

        Args:
            threshold (float) : cutoff threshold for probabilities to filter out low-probability errors (applied only to ECR gates).
                                Default is None, which means no filtering.

        Returns:
            tuple[dict, dict]: A tuple containing the single-qubit error instructions and ECR error instructions.
        """
        single_qubit_errors = self.extract_single_qubit_qerrors(
            self.noise_model["errors"],
            ["x", "sx", "rz"],
            self.use_qubits
        )
        ecr_errors = self.extract_ecr_errors(
            self.noise_model["errors"],
            self.use_qubits
        )
        single_qubit_error_instructions = self.post_process_single_qubit_errors(single_qubit_errors)
        ecr_error_instructions = self.post_process_ecr_errors(ecr_errors, threshold)
        return single_qubit_error_instructions, ecr_error_instructions