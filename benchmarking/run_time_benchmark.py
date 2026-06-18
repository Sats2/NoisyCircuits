import time
from NoisyCircuits import QuantumCircuit
import pickle
import matplotlib.pyplot as plt
import os
import json
import numpy as np

np.random.seed(42)


def density_matrix_simulation(circuit):
    return circuit.run_with_density_matrix(list(range(circuit.num_qubits)), 50)

def custom_simulation(circuit, trajectories):
    return circuit.execute(list(range(circuit.num_qubits)), trajectories, 50)

def generate_random_circuit(circuit, save_loc, depth=100):
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
    
def plot(data, save_loc, title):
    plt.figure(figsize=(10, 6))
    trajectory_list = [10, 100, 500, 1000]
    trajectory_times = {
        t : {"mean" : [], "std" : []} for t in trajectory_list
    }
    qubit_list = []
    density_matrix_times = {
        "mean" : [],
        "std" : []
    }
    for qubit in data.keys():
        for t in trajectory_list:
            trajectory_times[t]["mean"].append(np.mean(data[qubit]["MCWF"][t]) / 1e9)
            trajectory_times[t]["std"].append(np.std(data[qubit]["MCWF"][t]) / 1e9)
        density_matrix_times["mean"].append(np.mean(data[qubit]["Density_Matrix"]) / 1e9)
        density_matrix_times["std"].append(np.std(data[qubit]["Density_Matrix"]) / 1e9)
        qubit_list.append(qubit)
    
    plt.plot(qubit_list, density_matrix_times["mean"], label="Density Matrix")
    plt.fill_between(
        qubit_list,
        np.array(density_matrix_times["mean"]) - 2*np.array(density_matrix_times["std"]),
        np.array(density_matrix_times["mean"]) + 2*np.array(density_matrix_times["std"])
    )
    for t in trajectory_list:
        plt.plot(qubit_list, trajectory_times[t]["mean"], label="{} Trajectories".format(t))
        plt.fill_between(
            qubit_list,
            np.array(trajectory_times[t]["mean"]) - 2*np.array(trajectory_times[t]["std"]),
            np.array(trajectory_times[t]["mean"]) + 2*np.array(trajectory_times[t]["std"])
        )
    plt.xlabel("Number of Qubits")
    plt.ylabel("Simulation Runtime [s]")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(save_loc, dpi=600)
    plt.clf()

if __name__ == "__main__":
    max_qubits = 16
    noise_model_file = os.path.join(os.path.expanduser("~"), "benchmarking_data/Noise_Model_for_Benchmark.pkl")
    noise_model = pickle.load(open(noise_model_file, "rb"))
    data = {}
    trajectory_list =  [10, 100, 500, 1000]
    base_dir = os.path.join(os.path.expanduser("~"), "benchmarking_data/Time")
    num_trials = 40
    data_logger = open(os.path.join(base_dir, "Time_Benchmark_Consolidated_Output.txt"), "w")

    for qubits in range(2, max_qubits + 1):
        print("Running Benchmark for {} Qubits".format(qubits))
        qubit_dir = os.path.join(base_dir, f"{qubits}_Qubits")
        os.mkdir(qubit_dir)
        save_loc = os.path.join(qubit_dir, "Circuit_Instructions")
        os.mkdir(save_loc)
        circuit = QuantumCircuit(
            num_qubits = qubits,
            noise_model = noise_model,
            backend_qpu_type = "heron",
            sim_backend = "custom",
            threshold = 1e-6,
            verbose = False
        )
        qubit_data = {
            "MCWF" : {
                t : [] for t in trajectory_list
            },
            "Density_Matrix" : []
        }
        for trial in range(num_trials):
            save_file = os.path.join(save_loc, "Circuit_Instructions_{}_Qubits_{}_Trial.json".format(qubits, trial))
            generate_random_circuit(circuit, save_file)
            for trajectories in trajectory_list:
                t0 = time.perf_counter_ns()
                p = custom_simulation(circuit, trajectories)
                t1 = time.perf_counter_ns()
                qubit_data["MCWF"][trajectories].append(t1 - t0)
            t0 = time.perf_counter_ns()
            p = density_matrix_simulation(circuit)
            t1 = time.perf_counter_ns()
            qubit_data["Density_Matrix"].append(t1 - t0)
        with open(os.path.join(qubit_dir, "Raw_Benchmark_{}_Qubits.pkl".format(qubits)), "wb") as f:
            pickle.dump(qubit_data, f)
        data[qubits] = qubit_data
        data_logger.write("Qubits: {}\n".format(qubits))
        for trajectories in trajectory_list:
            times = qubit_data["MCWF"][trajectories]
            mean_time = np.mean(times)
            std_time = np.std(times)
            data_logger.write("Trajectory Number: {}\nMean Simulation Runtime: {} s, Std. Dev: {} s\n".format(trajectories, mean_time*1e-9, std_time*1e-9))
            print("Trajectory Number: {}\nMean Simulation Runtime: {} s, Std. Dev: {} s".format(trajectories, mean_time*1e-9, std_time*1e-9))
        dm_times = qubit_data["Density_Matrix"]
        mean_dm_time = np.mean(dm_times)
        std_dm_time = np.std(dm_times)
        data_logger.write("Density Matrix Simulation:\nMean Simulation Runtime: {} s, Std. Dev: {} s\n".format(mean_dm_time*1e-9, std_dm_time*1e-9))
        data_logger.write("-" * 50 + "\n\n")
        data_logger.flush()
        print("Density Matrix Simulation:\nMean Simulation Runtime: {} s, Std. Dev: {} s".format(mean_dm_time*1e-9, std_dm_time*1e-9))
        print("Completed Benchmark for {} Qubits".format(qubits))
        print("-" * 50)
    
    with open(os.path.join(base_dir, "Full_Time_Benchmark_Data.pkl"), "wb") as f:
        pickle.dump(data, f)
    data_logger.close()
    fig_save_loc = os.path.join(base_dir, "Time_Benchmark_Plot.svg")
    plot(data, fig_save_loc, "Runtime Comparison: Density Matrix Vs MCWF")