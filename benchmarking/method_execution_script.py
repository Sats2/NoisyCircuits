from NoisyCircuits import QuantumCircuit
import numpy as np
import sys
import pickle
import os
import json

def density_matrix_simulation(circuit):
    p = circuit.run_with_density_matrix(list(range(circuit.num_qubits)), 50)
    return p

def custom_simulation(circuit, trajectories):
    p = circuit.execute(list(range(circuit.num_qubits)), trajectories, 50)
    return p

if __name__ == "__main__":
    noise_model_file = os.path.join(os.path.expanduser("~"), "benchmarking_data/Noise_Model_for_Benchmark.pkl")
    noise_model = pickle.load(open(noise_model_file, "rb"))
    input_args = sys.argv[1:]
    qubits, json_location, run_density_matrix, trajectories = int(input_args[0]), input_args[1], input_args[2].lower() == "true", int(input_args[3])
    circuit = QuantumCircuit(
                num_qubits = qubits,
                noise_model = noise_model,
                backend_qpu_type = "heron",
                sim_backend = "custom",
                threshold = 1e-8,
                verbose = False
            )
    with open(json_location, "r") as f:
        instruction_list = json.load(f)
    circuit.instruction_list = instruction_list
    if run_density_matrix:
        p = density_matrix_simulation(circuit)
    else:
        p = custom_simulation(circuit, trajectories)