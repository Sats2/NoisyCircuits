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
            ops_list = [
                np.array([[complex(*e) for e in row] for row in matrix])
                for matrix in inst["params"]
            ]
            ops_list = [op.astype(np.complex128) for op in ops_list]
            return ops_list
    return None

class BuildModel:
    def __init__(self,
                 noise_model:dict,
                 num_qubits:int,
                 threshold:float=None):
        self.noise_model = noise_model
        self.num_qubits = num_qubits
        self.use_qubits = list(range(num_qubits))
        self.max_ancilla = 0
        self.threshold = threshold

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
    
    def build_kraus_unitary(self, kraus_ops:list)->np.ndarray:
        """
        Builds a unitary operator from the given Kraus operators.

        Args:
            kraus_ops (list): List of Kraus operators.  

        Returns:
            np.ndarray: Unitary operator constructed from the Kraus operators.
        """
        self._ensure_ctpt(kraus_ops)
        dim_sys = 2
        num_kraus = len(kraus_ops)
        dim_env = 2 ** (math.ceil(np.log2(num_kraus)))
        V = np.vstack(kraus_ops)
        target_rows = dim_sys * dim_env
        current_rows = V.shape[0]
        pad_rows = target_rows - current_rows
        if pad_rows > 0:
            V_padded = np.vstack([V, np.zeros((pad_rows, dim_sys), dtype=np.complex128)])
        else:
            V_padded = V
        Q, _ = np.linalg.qr(V_padded, mode="complete")
        U_springstine = Q
        return U_springstine
    
    def _drop_errors(self,
                     probabilities:list,
                     instructions:list,
                     threshold:float)->tuple[list, list]:
        probs = np.array(probabilities)
        instrs = np.array(instructions)
        mask = probs >= threshold
        filtered_probs = probs[mask]
        filtered_instrs = instrs[mask]
        filtered_probs /= np.sum(filtered_probs)
        filtered_instrs = list(map(str, filtered_instrs))
        filtered_probs = list(map(float, filtered_probs))
        return filtered_probs, filtered_instrs
    
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
        single_qubit_errors_processed = {}
        instruction_map = {
            "id" : np.array([[1, 0], [0, 1]]),
            "x" : np.array([[0, 1], [1, 0]]),
            "y" : np.array([[0, -1j], [1j, 0]]),
            "z" : np.array([[1, 0], [0, -1]]),
            "K0" : np.array([[1, 0], [0, 0]]),
            "K1" : np.array([[0, 1], [0, 0]])
        }
        for qubit in single_qubit_errors.keys():
            qubit_errors = {}
            for gate in single_qubit_errors[qubit].keys():
                instructions = single_qubit_errors[qubit][gate]["instructions"]
                probabilities = single_qubit_errors[qubit][gate]["probabilities"]
                kraus_ops = single_qubit_errors[qubit][gate]["kraus"]
                if self.threshold is not None:
                    probabilities, instructions = self._drop_errors(
                                                        probabilities, instructions, self.threshold
                                                        )
                operators = []
                ops = None
                use_ops = None
                for instr, prob in zip(instructions, probabilities):
                    instruction = instr.split("-")
                    op = np.eye(2, dtype=np.complex128)
                    for name in instruction:
                        if name not in ["reset", "kraus"]:
                            op = np.dot(instruction_map[name], op)
                        elif name == "reset":
                            op1 = np.dot(instruction_map["K0"], op)
                            op2 = np.dot(instruction_map["K1"], op)
                            ops = [op1, op2]
                        elif name == "kraus":
                            if ops is not None:
                                use_ops = []
                                for kop in kraus_ops:
                                    for op in ops:
                                        use_op = np.dot(kop, op)
                                        use_ops.append(use_op)
                            else:
                                use_ops = []
                                for kop in kraus_ops:
                                    use_op = np.dot(kop, op)
                                    use_ops.append(use_op)
                        if use_ops is not None:
                            for op in use_ops:
                                operators.append(op * prob)
                        else:
                            operators.append(op * prob)
                unitary_noise_gate = self.build_kraus_unitary(operators)
                num_ancilla_needed = math.ceil(np.log2(len(operators)))
                self.max_ancilla = max(self.max_ancilla, num_ancilla_needed)
                qubit_errors[gate] = {
                    "unitary" : unitary_noise_gate,
                    "num_ancilla" : num_ancilla_needed,
                    "instructions" : instructions,
                    "probabilities" : probabilities,
                    "kraus" : kraus_ops
                }
            single_qubit_errors_processed[qubit] = qubit_errors
        return single_qubit_errors_processed

    # TODO: Complete the implementation for single/double instructions with/without Kraus operators.
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
        instruction_map = {
            "id" : np.array([[1,0], [0,1]]),
            "x" : np.array([[0,1], [1,0]]),
            "y" : np.array([[0,-1j], [1j,0]]),
            "z" : np.array([[1,0], [0,-1]]),
            "K0" : np.array([[1,0], [0,0]]),
            "K1" : np.array([[0,1], [0,0]])
        }
        for qubit_pair in ecr_error.keys():
            for item in ecr_error[qubit_pair]:
                qubit_1 = qubit_pair[0]
                qubit_2 = qubit_pair[1]
                instructions = item["instructions"]
                probabilities = item["probability"]
                for error in instructions:
                    if error["name"] == "pauli":
                        pass
                    elif error["name"] == "kraus":
                        pass
                                    

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