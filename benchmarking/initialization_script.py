import numpy as np
import json
import sys

np.random.seed(42)


def build_random_circuit(num_qubits, depth):
    instruction_list = []
    single_gates = [
        "x",
        "rx",
        "rz",
        "sx"
    ]
    double_gates = [
        "cz",
        "rzz"
    ]
    for _ in range(depth):
        for q in range(num_qubits):
            gate = np.random.choice(single_gates)
            param = np.random.uniform(-2*np.pi, 2*np.pi)
            instruction_list.append([gate, [q, q], param])
        for q in range(num_qubits-1):
            gate = np.random.choice(double_gates)
            param = np.random.uniform(-2*np.pi, 2*np.pi)
            instruction_list.append([gate, [q, q+1], param])
    return instruction_list

if __name__ == "__main__":
    input_args = sys.argv[1:]
    qubits = int(input_args[0])
    depth = int(input_args[1])
    instruction_list = build_random_circuit(qubits, depth=depth)
    save_loc = input_args[2]
    with open(save_loc, "w") as f:
        json.dump(instruction_list, f)