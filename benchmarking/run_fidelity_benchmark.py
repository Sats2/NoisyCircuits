import os
from NoisyCircuits import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


np.random.seed(42)

def compute_fidelity_metrics(p_comp, p_ref):
    bhattacharyya_coeff = np.sum(np.sqrt(p_comp * p_ref))
    bhattacharrya_distance = -np.log(bhattacharyya_coeff)
    hellinger_distance = np.sqrt(1 - bhattacharyya_coeff)
    m = 0.5 * (p_comp + p_ref)
    js_divergence = 0.5 * (np.sum(p_comp * np.log(p_comp / m)) + np.sum(p_ref * np.log(p_ref / m)))
    return bhattacharyya_coeff, bhattacharrya_distance, hellinger_distance, js_divergence

def run_custom_simulation(circuit, trajectories):
    return circuit.execute(list(range(circuit.num_qubits)), trajectories, 50)

def run_density_matrix_simulation(circuit):
    return circuit.run_with_density_matrix(list(range(circuit.num_qubits)), 50)

def generate_random_circuit(circuit, save_loc, depth):
    circuit.refresh()
    single_qubit_gates = {
        "x" : lambda p, q: circuit.X(q),
        "sx" : lambda p, q: circuit.SX(q),
        "rz" : lambda p, q: circuit.RZ(p, q),
        "rx" : lambda p, q: circuit.RX(p, q)
    }
    two_qubit_gates = {
        "cz" : lambda p, q: circuit.CZ(q[0], q[1]),
        "rzz" : lambda p, q: circuit.RZZ(p, q[0], q[1])
    }
    for _ in range(depth):
        for q in range(circuit.num_qubits):
            gate = np.random.choice(list(single_qubit_gates.keys()))
            single_qubit_gates[gate](np.random.uniform(-2*np.pi, 2*np.pi), q)
        for q in range(circuit.num_qubits - 1):
            gate = np.random.choice(list(two_qubit_gates.keys()))
            two_qubit_gates[gate](np.random.uniform(-2*np.pi, 2*np.pi), [q, q+1])
    with open(save_loc, "w") as f:
        json.dump(circuit.instruction_list, f)

def plot():
    pass

if __name__ == "__main__":
    noise_model_loc = os.path.join(os.path.expanduser("~"), "benchmarking_data/Noise_Model_for_Benchmark.pkl")
    noise_model = pickle.load(open(noise_model_loc, "rb"))
    trajectory_list = [10, 100, 500, 1000]
    max_qubits = 16
    num_trials = 100
    data = {}

    base_dir = os.path.join(os.path.expanduser("~"), "benchmarking_data/Fidelity")
    data_logger = open(os.path.join(base_dir, "Fidelity_Benchmark_Consolidated_Results.txt"), "w")

    for qubits in range(2, max_qubits + 1):
        data_logger.write(f"Running fidelity benchmark for {qubits} Qubits.\n")
        qubit_data = {}
        qubit_loc = os.path.join(base_dir, f"{qubits}_Qubits")
        os.mkdir(qubit_loc)

        circuit = QuantumCircuit(
            num_qubits = qubits,
            noise_model = noise_model,
            backend_qpu_type = "heron",
            sim_backend = "custom",
            threshold = 1e-6,
            verbose = False
        )

        for depth in [1, 10, 100, 200]:
            depth_loc = os.path.join(qubit_loc, f"{depth}_Depth")
            os.mkdir(depth_loc)

            circuit_loc = os.path.join(depth_loc, "Circuit_Instructions")
            os.mkdir(circuit_loc)

            depth_data = {
                t : {
                    "BC" : [],
                    "BD" : [],
                    "HD" : [],
                    "JS" : []
                } 
                for t in trajectory_list
            }

            for trial in range(num_trials):
                circuit_file_name = os.path.join(circuit_loc, "Circuit_Instructions_{}_Qubits_{}_Trial.json".format(qubits, trial))
                generate_random_circuit(circuit, circuit_file_name, depth)

                p_ref = run_density_matrix_simulation(circuit)

                for trajectories in trajectory_list:
                    p_comp = run_custom_simulation(circuit, trajectories)
                    metrics = compute_fidelity_metrics(p_comp, p_ref)