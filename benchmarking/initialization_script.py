import pickle
from NoisyCircuits import QuantumCircuit
import os
import numpy as np
import json
import sys

np.random.seed(42)

noise_model_file = os.path.join(os.path.expanduser("~"), "benchmarking_data/Noise_Model_for_Benchmark.pkl")
noise_model = pickle.load(open(noise_model_file, "rb"))


def build_random_circuit(circuit, depth):
    instruction_list = []
    single_gates = {
        "x" : lambda p, q: circuit.X(q),
        "sx" : lambda p, q: circuit.SX(q),
        "rz" : lambda p, q: circuit.RZ(p, q),
        "rx" : lambda p, q: circuit.RX(p, q)
    }
    double_gates = {
        "cz" : lambda p, q1, q2: circuit.CZ(q1, q2),
        "rzz" : lambda p, q1, q2: circuit.RZZ(p, q1, q2)
    }
    for _ in range(depth):
        for i in range(circuit.num_qubits):
            gate_name = np.random.choice(list(single_gates.keys()))
            param = np.random.uniform(-2*np.pi, 2*np.pi)
            single_gates[gate_name](param, i)
            instruction_list.append([gate_name, [i, i], param])
        for q in range(circuit.num_qubits - 1):
            gate_name = np.random.choice(list(double_gates.keys()))
            param = np.random.uniform(-2*np.pi, 2*np.pi)
            double_gates[gate_name](param, q, q + 1)
            instruction_list.append([gate_name, [q, q+1], param])
    return instruction_list

if __name__ == "__main__":
    input_args = sys.argv[1:]
    qubits = int(input_args[0])
    depth = int(input_args[1])
    circuit = QuantumCircuit(
                num_qubits = qubits,
                noise_model = noise_model,
                backend_qpu_type = "heron",
                sim_backend = "custom",
                threshold = 1e-8,
                verbose = False
            )
    circuit.refresh()
    instruction_list = build_random_circuit(circuit, depth=depth)
    save_loc = input_args[2]
    with open(save_loc, "w") as f:
        json.dump(instruction_list, f)