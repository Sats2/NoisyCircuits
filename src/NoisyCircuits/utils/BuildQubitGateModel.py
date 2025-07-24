from pennylane import numpy as np
from collections import defaultdict
import math

def _map_instruction_qubits(instrs, gate_qubits)->list:
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

def _extract_kraus(instrs)->list:
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
                 threshold:float=1e-12):
        """
        Initializes the BuildModel with a noise model, number of qubits, and an optional threshold.

        Args:
            noise_model (dict): The noise model to use. Provided as a JSON-ised dictionary.
            num_qubits (int): The number of qubits in the model.
            threshold (float, optional): The threshold for noise. Defaults to None.
        
        Raises:
            TypeError: If noise_model is not a dictionary or num_qubits is not an integer.
            ValueError: If num_qubits is not a positive integer or threshold is not between 0 and 1.
        """
        if not isinstance(noise_model, dict):
            raise TypeError("Noise model must be a dictionary.")
        if not isinstance(num_qubits, int):
            raise TypeError("Number of qubits must be an integer.")
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        if not isinstance(threshold, (float, int)):
            raise TypeError("Threshold must be a float or int.")
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1.")
        self.noise_model = noise_model
        self.num_qubits = num_qubits
        self.use_qubits = list(range(num_qubits))
        self.threshold = threshold

    def _ensure_ctpt(self,
                    kraus_ops:list)->bool:
        """Ensures that the given Kraus operators form a CPTP map.

        Args:
            kraus_ops (list): List of Kraus operators to check.

        Raises:
            Warning: If the Kraus operators do not form a CPTP map.

        Returns:
            bool: True if the Kraus operators form a CPTP map, False otherwise.
        """
        mat = np.zeros(kraus_ops[0].shape, dtype=complex, requires_grad=False)
        for op in kraus_ops:
            mat += np.dot(op.conj().T, op)
        return np.allclose(mat, np.eye(mat.shape[0], dtype=complex), atol=1e-6)
    
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
    
    def build_full_matrix(self,
                          P:np.ndarray,
                          n:int, 
                          system:int)->np.ndarray:
        """
        Extends a single-qubit operator P to a full system operator by applying P to the nth qubit and identity to all others.

        Args:
            P (np.ndarray): The single-qubit operator to extend.
            n (int): The index of the qubit to which P is applied.
            system (int): The total number of qubits in the system.

        Returns:
            np.ndarray: The full system operator.
        """
        op_list = []
        for i in range(system):
            if i == n:
                op_list.append(P)
            else:
                op_list.append(np.eye(2, requires_grad=False))
        full_op = op_list[0]
        for op in op_list[1:]:
            full_op = np.kron(full_op, op)
        return full_op
    
    def extend_kraus_to_system(self, 
                               kraus_ops:list, 
                               qubit_idx:int)->list:
        """Extends Kraus operators to the full system.

        Args:
            kraus_ops (list): List of Kraus operators for the subsystem.
            qubit_idx (int): Index of the qubit to which the operators are applied.

        Returns:
            list: List of extended Kraus operators for the full system.
        """
        system_qubits = self.num_qubits
        extended_kraus = []
        for op in kraus_ops:
            extended_op = self.build_full_matrix(op, qubit_idx, system_qubits)
            extended_kraus.append(extended_op)
        return extended_kraus
    
    def build_full_matrix_2qubit(self,
                                 P:np.ndarray,
                                 qubit_pair:tuple,
                                 system:int)->np.ndarray:
        """
        Extends a 2 qubit operator P to a full system operator by applying P to the specified qubit pair and identity to all others.

        Args:
            P (np.ndarray): The two-qubit operator to extend (4x4 matrix).
            qubit_pairt (tuple): Tuple of qubit indices to which P is applied.
            system (int): Total number of qubits in the system.

        Returns:
            np.ndarray: The full system operator (2^system x 2^system matrix). 
        """
        q0, q1 = qubit_pair
        dim = 2**system
        full_op = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            for j in range(dim):
                state_i = [(i >> k) & 1 for k in range(system)]
                state_j = [(j >> k) & 1 for k in range(system)]
                i_q0, i_q1 = state_i[q0], state_i[q1]
                j_q0, j_q1 = state_j[q0], state_j[q1]
                other_qubits_same = all(state_i[k] == state_j[k] for k in range(system) if k not in [q0, q1])
                if other_qubits_same:
                    i_2q = i_q0 * 2 + i_q1
                    j_2q = j_q0 * 2 + j_q1
                    full_op[i,j] = P[i_2q, j_2q]
        return full_op

    def extend_kraus_to_system_multiqubit(self, kraus_ops:list, qubit_pair:tuple)->list:
        """
        Extends Kraus operators for a two-qubit gate to the full system.

        Args:
            kraus_ops (list): List of Kraus operators for the two-qubit gate.
            qubit_pair (tuple): Tuple of qubit indices for which the Kraus operators are defined.

        Returns:
            list: List of extended Kraus operators for the full system.
        """
        system_qubits = self.num_qubits
        extended_kraus = []
        for op in kraus_ops:
            extended_op = self.build_full_matrix_2qubit(op, qubit_pair, system_qubits)
            extended_kraus.append(extended_op)
        return extended_kraus        
    
    def _drop_errors(self,
                     probabilities:list,
                     instructions:list,
                     threshold:float)->tuple[list, list]:
        """
        Filters out errors based on a probability threshold for single qubit error models.

        Args:
            probabilities (list): List of error probabilities.
            instructions (list): List of error instructions.
            threshold (float): Probability threshold for filtering.

        Returns:
            tuple[list, list]: Filtered probabilities and instructions.
        """
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
                    assert len(probabilities) == len(instructions), "Probabilities and instructions must have the same length after filtering."
                    assert np.allclose(np.sum(probabilities), 1), "Filtered probabilities must sum to 1."
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
                            operators.append(np.sqrt(prob) * op)
                    else:
                        operators.append(np.sqrt(prob) * op)
                if not self._ensure_ctpt(operators):
                    print(f"Warning: Original Kraus operators for qubit {qubit} do not form a CPTP map.")
                    mat = np.zeros((2,2), dtype=complex)
                    for op in operators:
                        mat += np.dot(op.conj().T, op)
                    print(mat)
                kraus_operators = self.extend_kraus_to_system(operators, qubit)
                if not self._ensure_ctpt(kraus_operators):
                    print(f"Warning: Extended Kraus operators for qubit {qubit} do not form a CPTP map.")
                qubit_errors[gate] = {
                    "kraus_operators" : kraus_operators,
                    "instructions" : instructions,
                    "probabilities" : probabilities,
                    "kraus" : kraus_ops
                }
            single_qubit_errors_processed[qubit] = qubit_errors
        return single_qubit_errors_processed

    def post_process_ecr_errors(self,
                                ecr_errors:dict,
                                threshold:float=1e-12)->dict:
        """
        Post-processes the ECR errors to create a dictionary of instructions, probabilities, and Kraus operators for each qubit pair.

        Args:
            ecr_errors (dict): The ECR errors to process.
            threshold (float, optional): The probability threshold for filtering errors. Defaults to 1e-12.

        Returns:
            dict: A dictionary containing the processed ECR errors.
        """
        processed_errors = {}
        print("Processing ECR errors.")
        for qpair, error_list in ecr_errors.items():
            instructions = []
            probabilities = []
            kraus_ops = []
            kraus_qubits = set()
            unique_kraus_per_qubit = {}  # qubit -> unique kraus operators
            filtered_count = 0
            total_count = len(error_list)
            for error_entry in error_list:
                prob = error_entry["probability"]
                if prob < threshold:
                    filtered_count += 1
                    continue
                probabilities.append(prob)
                pauli_instruction = None
                pauli_qubits = None
                gate_instructions = {}
                for inst in error_entry["instructions"]:
                    inst_name = inst["name"]
                    inst_qubits = inst.get("qubits", [])
                    if inst_name == "pauli":
                        params = inst.get("params", ["II"])
                        pauli_instruction = params[0] if params else "II"
                        pauli_qubits = inst_qubits
                    elif inst_name == "kraus":
                        kraus_params = inst.get("params", [])
                        if kraus_params and inst_qubits:
                            qubit = inst_qubits[0]
                            if qubit not in unique_kraus_per_qubit:
                                unique_kraus_per_qubit[qubit] = kraus_params
                                kraus_qubits.add(qubit)
                        for q in inst_qubits:
                            gate_instructions[q] = inst_name
                    else:
                        for q in inst_qubits:
                            gate_instructions[q] = inst_name
                final_instruction = []
                qubits_list = list(qpair)
                if pauli_instruction and pauli_qubits:
                    pauli_map = {}
                    for i, q in enumerate(pauli_qubits):
                        if i < len(pauli_instruction):
                            pauli_map[q] = pauli_instruction[i]
                    reordered_pauli = ""
                    for q in qubits_list:
                        reordered_pauli += pauli_map.get(q, "I")
                    final_instruction.append(reordered_pauli)
                gate_parts = []
                for q in qubits_list:
                    gate_name = gate_instructions.get(q, "id")
                    gate_parts.append(gate_name)
                if gate_parts:
                    combined_gate = "-".join(gate_parts)
                    final_instruction.append(combined_gate)
                final_instruction.append(qubits_list)
                instructions.append(final_instruction)
                kraus_ops.append(None)
            if instructions:
                if unique_kraus_per_qubit:
                    unique_kraus_per_qubit = [_extract_kraus(inst) for inst in unique_kraus_per_qubit.values()]
                    unique_kraus_operators = {q: ops for q, ops in zip(unique_kraus_per_qubit.keys(), unique_kraus_per_qubit)}
                    probabilities = np.array(probabilities)
                    probabilities /= np.sum(probabilities)
                else:
                    unique_kraus_operators = None
                processed_errors[qpair] = {
                    "instructions": instructions,
                    "probabilities": probabilities,
                    "kraus": unique_kraus_operators,
                    "kraus_qubit": list(kraus_qubits)
                }
                print(f"  Qubit pair {qpair}: {len(instructions)}/{total_count} errors above threshold "
                  f"({filtered_count} filtered out)")
        print("ECR errors processed.")
        return processed_errors
        
    def get_ecr_noise_operators(self, ecr_errors:dict)->dict:
        ecr_error_operators = {}
        instruction_map = {
            "id" : np.array([[1, 0], [0, 1]]),
            "x" : np.array([[0, 1], [1, 0]]),
            "y" : np.array([[0, -1j], [1j, 0]]),
            "z" : np.array([[1, 0], [0, -1]]),
            "K0" : np.array([[1, 0], [0, 0]]),
            "K1" : np.array([[0, 1], [0, 0]]),
            "i" : np.array([[1, 0], [0, 1]])
        }                                   
        for qpair, error_data in ecr_errors.items():
            full_instructions = error_data["instructions"]
            probabilities = error_data["probabilities"]
            probabilities = np.array(probabilities)
            probabilities /= np.sum(probabilities)
            probabilities = probabilities.tolist()
            kraus_ops = error_data["kraus"]
            error_operators = []
            for instruction,prob in zip(full_instructions, probabilities):
                qubits = instruction[-1]
                major_inst = instruction[0]
                minor_inst = instruction[1]
                q0_ops = instruction_map[major_inst[0].lower()]
                q1_ops = instruction_map[major_inst[1].lower()]
                minor_ops = minor_inst.split("-")
                if "kraus" not in minor_ops:
                    if "reset" not in minor_ops:
                        q0_ops = np.dot(instruction_map[minor_ops[0]], q0_ops)
                        q1_ops = np.dot(instruction_map[minor_ops[1]], q1_ops)
                        k_op = np.kron(q0_ops, q1_ops)
                        error_operators.append(np.sqrt(prob) * k_op)
                    else:
                        if minor_ops[0] == "reset" and minor_ops[1] == "reset":
                            q0_ops1 = np.dot(instruction_map["K0"], q0_ops)
                            q0_ops2 = np.dot(instruction_map["K1"], q0_ops)
                            q1_ops1 = np.dot(instruction_map["K0"], q1_ops)
                            q1_ops2 = np.dot(instruction_map["K1"], q1_ops)
                            k_op1 = np.kron(q0_ops1, q1_ops1)
                            k_op2 = np.kron(q0_ops2, q1_ops2)
                            k_op3 = np.kron(q0_ops1, q1_ops2)
                            k_op4 = np.kron(q0_ops2, q1_ops1)
                            error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2, k_op3, k_op4])
                        elif minor_ops[0] == "reset":
                            q0_op1 = np.dot(instruction_map["K0"], q0_ops)
                            q0_op2 = np.dot(instruction_map["K1"], q0_ops)
                            q1_ops = np.dot(instruction_map[minor_ops[1]], q1_ops)
                            k_op1 = np.kron(q0_op1, q1_ops)
                            k_op2 = np.kron(q0_op2, q1_ops)
                            error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
                        else:
                            q0_ops = np.dot(instruction_map[minor_ops[0]], q0_ops)
                            q1_op1 = np.dot(instruction_map["K0"], q1_ops)
                            q1_op2 = np.dot(instruction_map["K1"], q1_ops)
                            k_op1 = np.kron(q0_ops, q1_op1)
                            k_op2 = np.kron(q0_ops, q1_op2)
                            error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
                else:
                    if minor_ops[0] == "kraus" and minor_ops[1] == "kraus":
                        kraus_ops_for_q0 = kraus_ops[qubits[0]]
                        kraus_ops_for_q1 = kraus_ops[qubits[1]]
                        kraus_operations_for_q0 = []
                        kraus_operations_for_q1 = []
                        for kraus_op in kraus_ops_for_q0:
                            kraus_operations_for_q0.append(np.dot(kraus_op, q0_ops))
                        for kraus_op in kraus_ops_for_q1:
                            kraus_operations_for_q1.append(np.dot(kraus_op, q1_ops))
                        for kraus_op0 in kraus_operations_for_q0:
                            for kraus_op1 in kraus_operations_for_q1:
                                k_op = np.kron(kraus_op0, kraus_op1)
                                error_operators.append(np.sqrt(prob) * k_op)
                    elif minor_ops[0] == "kraus":
                        kraus_ops_for_q0 = kraus_ops[qubits[0]]
                        kraus_operations_for_q0 = []
                        for kraus_op in kraus_ops_for_q0:
                            kraus_operations_for_q0.append(np.dot(kraus_op, q0_ops))
                        if minor_ops[1] == "reset":
                            q1_op1 = np.dot(instruction_map["K0"], q1_ops)
                            q1_op2 = np.dot(instruction_map["K1"], q1_ops)
                            for kraus_op in kraus_operations_for_q0:
                                k_op1 = np.kron(kraus_op, q1_op1)
                                k_op2 = np.kron(kraus_op, q1_op2)
                                error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
                        else:
                            q1_ops = np.dot(instruction_map[minor_ops[1]], q1_ops)
                            for kraus_op in kraus_operations_for_q0:
                                k_op = np.kron(kraus_op, q1_ops)
                                error_operators.append(np.sqrt(prob) * k_op)
                    else:
                        kraus_ops_for_q1 = kraus_ops[qubits[1]]
                        kraus_operations_for_q1 = []
                        for kraus_op in kraus_ops_for_q1:
                            kraus_operations_for_q1.append(np.dot(kraus_op, q1_ops))
                        if minor_ops[0] == "reset":
                            q0_op1 = np.dot(instruction_map["K0"], q0_ops)
                            q0_op2 = np.dot(instruction_map["K1"], q0_ops)
                            for kraus_op in kraus_operations_for_q1:
                                k_op1 = np.kron(q0_op1, kraus_op)
                                k_op2 = np.kron(q0_op2, kraus_op)
                                error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
                        else:
                            q0_ops = np.dot(instruction_map[minor_ops[0]], q0_ops)
                            for kraus_op in kraus_operations_for_q1:
                                k_op = np.kron(q0_ops, kraus_op)
                                error_operators.append(np.sqrt(prob) * k_op)
            error_operators_full_system = self.extend_kraus_to_system_multiqubit(error_operators, qpair)
            if not self._ensure_ctpt(error_operators_full_system):
                print(f"Warning: Kraus operators for qubit pair {qpair} do not form a CPTP map.")
            ecr_error_operators[qpair] = {
                "operators": error_operators_full_system
            }
        return ecr_error_operators

    def _create_connectivity_map(self, ecr_error_instructions:dict, use_qubits:list)->dict:
        """
        Creates a connectivity map from ECR error instructions showing which qubits are connected.

        Args:
            ecr_error_instructions (dict): Dictionary of ECR error instructions with qubit pairs as keys.
            use_qubits (list): List of qubits being used in the system.
        
        Returns:
            dict: Connectivity map showing which qubits are connected to each other.
        """
        connectivity_map = {qubit: [] for qubit in use_qubits}
        
        # Extract connectivity from ECR error instruction keys (qubit pairs)
        for qubit_pair in ecr_error_instructions.keys():
            if isinstance(qubit_pair, tuple) and len(qubit_pair) == 2:
                q0, q1 = qubit_pair
                if q0 in use_qubits and q1 in use_qubits:
                    if q1 not in connectivity_map[q0]:
                        connectivity_map[q0].append(q1)
                    if q0 not in connectivity_map[q1]:
                        connectivity_map[q1].append(q0)
        
        return connectivity_map

    def extract_measurement_errors(self,
                                    data:list,
                                    extract_qubits:list[int])->dict:
        """
        Extracts the measurement errors from the noise model data.

        Args:
            data (list): List of Noise model entries from Qiskit JSON.
            extract_qubits (list[int]): List of qubits for which to extract measurement errors.
        
        Returns:
            dict: Dictionary of measurement errors for the specified qubits.
        """
        roerror_map = {}
        for entry in data:
            if entry.get("type") != "roerror":
                continue
            operations = entry.get("operations", [])  # Fixed typo: was "gat_qubits"
            target_ops = ["measure"]
            if target_ops and not any(op in target_ops for op in operations):
                continue
            gate_qubits_list = entry.get("gate_qubits", []) 
            prob_matrices = entry.get("probabilities", [])
            for gate_qubits, matrix in zip(gate_qubits_list, prob_matrices):
                if len(gate_qubits) == 1:
                    qubit = gate_qubits[0]
                    roerror_map[qubit] = {
                        "operation": "measure",
                        "matrix": matrix
                    }
        measurement_errors = {}
        print("Available qubits in roerror_map:", list(roerror_map.keys()))
        print("Requested qubits:", extract_qubits)
        for qubit in extract_qubits:
            if qubit not in roerror_map:
                print(f"Warning: No measurement error data found for qubit {qubit}. Using identity matrix.")
                # Use identity matrix (no error) if measurement error data is not available
                matrix = np.eye(2)
            else:
                matrix = np.zeros((2,2))
                matrix[0,0] = roerror_map[qubit]["matrix"][0]
                matrix[0,1] = roerror_map[qubit]["matrix"][1]
                matrix[1,0] = roerror_map[qubit]["matrix"][1]
                matrix[1,1] = roerror_map[qubit]["matrix"][0]
            measurement_errors[qubit] = matrix
        return measurement_errors

    def build_qubit_gate_model(self,
                               threshold:float=1e-12)->tuple[dict, dict, dict]:
        """
        Builds the qubit gate model by extracting single-qubit and ECR errors from the noise model.

        Args:
            threshold (float) : cutoff threshold for probabilities to filter out low-probability errors (applied only to ECR gates).
                                Default is None, which means no filtering.

        Returns:
            tuple[dict, dict, dict]: A tuple containing the single-qubit error instructions, ECR error instructions, and measurement error instructions.
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
        print("Completed Extraction of ECR Errors.\nStarting post-processing on Single Qubit Errors.")
        single_qubit_error_instructions = self.post_process_single_qubit_errors(single_qubit_errors)
        print("Completed post-processing on Single Qubit Errors.")
        ecr_error_post_processed = self.post_process_ecr_errors(ecr_errors, self.threshold)
        print("Building Noise Operators for ECR Errors.")
        ecr_error_instructions = self.get_ecr_noise_operators(ecr_error_post_processed)
        print("Completed building Noise Operators for ECR Errors.\nExtracting Measurement Errors.")
        measurement_errors = self.extract_measurement_errors(self.noise_model["errors"],
                                                              self.use_qubits)
        print("Completed Extraction of Measurement Errors.")
        print("Preparing Qubit Connectivity Map for Requested Qubits")
        connectivity_map = self._create_connectivity_map(ecr_error_instructions, self.use_qubits)
        print("Qubit Connectivity Map Prepared.")
        print("Returning Single Qubit Error Instructions, ECR Error Instructions, Measurement Errors and Connectivity Map.")
        return single_qubit_error_instructions, ecr_error_instructions, measurement_errors, connectivity_map