import os
from NoisyCircuits import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import json
import pickle


np.random.seed(42)

def compute_fidelity_metrics(p_comp, p_ref):
    bhattacharyya_coeff = np.sum(np.sqrt(p_comp * p_ref))
    bhattacharrya_distance = -np.log(bhattacharyya_coeff)
    hellinger_distance = np.sqrt(1 - bhattacharyya_coeff)
    js_divergence = jensenshannon(p_comp, p_ref)**2
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

def plot(data, depth, metric, save_loc):
    plt.figure(figsize=(10,6))
    trajectory_list = [10, 100, 500, 1000]
    metric_data = {
        t : {
            "mean" : [],
            "std" : []
        } for t in trajectory_list
    }
    qubit_list = []
    for qubit in data.keys():
        qubit_list.append(int(qubit.split("_")[0]))
        for trajectories in trajectory_list:
            fidelity_data = data[qubit][f"Depth_{depth}"][trajectories][metric]
            metric_data[trajectories]["mean"].append(np.mean(fidelity_data))
            metric_data[trajectories]["std"].append(np.std(fidelity_data))
            print("Qubits: {}\tTrajectories: {}\t{}: {:.4f} ± {:.4f}".format(qubit, trajectories, metric, np.mean(fidelity_data), np.std(fidelity_data)))
    for trajectories in trajectory_list:
        plt.plot(qubit_list, metric_data[trajectories]["mean"], label="{} Trajectories".format(trajectories))
        plt.fill_between(
            qubit_list,
            np.array(metric_data[trajectories]["mean"]) - 2*np.array(metric_data[trajectories]["std"]),
            np.array(metric_data[trajectories]["mean"]) - 2*np.array(metric_data[trajectories]["std"])
        )
    plt.xlabel("Number of Qubits")
    plt.ylabel(metric.replace("_", " "))
    plt.title("Fidelity between Density Matrix and MCWF Simulations at Depth {}".format(depth))
    plt.legend()
    plt.savefig(save_loc)
    plt.close()

if __name__ == "__main__":
    noise_model_loc = os.path.join(os.path.expanduser("~"), "benchmarking_data/Noise_Model_for_Benchmark.pkl")
    noise_model = pickle.load(open(noise_model_loc, "rb"))
    trajectory_list = [10, 100, 500, 1000]
    max_qubits = 16
    num_trials = 100
    data = {}

    base_dir = os.path.join(os.path.expanduser("~"), "benchmarking_data/Fidelity")
    data_logger = open(os.path.join(base_dir, "Fidelity_Benchmark_Consolidated_Results.txt"), "a")

    # data_logger.write("Qubits\t|\tDepth\t|\tTrajectories\t|\tBhattacharyya Coefficient\t|\tBhattacharyya Distance\t|\tHellinger Distance\t|\tJS Divergence\n")
    # data_logger.flush()

    for qubit in range(2, max_qubits+1):

        print(f"Running benchmark for {qubit} Qubits.")

        qubit_data = {}

        qubit_loc = os.path.join(base_dir, f"{qubit}_Qubits")
        if os.path.exists(qubit_loc):
            print(f"Directory for {qubit} Qubits exists.")
            continue

        os.mkdir(qubit_loc)

        circuit = QuantumCircuit(
            num_qubits = qubit,
            noise_model = noise_model,
            backend_qpu_type = "heron",
            sim_backend = "custom",
            threshold = 1e-8,
            verbose = False
        )

        for depth in [1, 10, 50, 100, 200]:
            print("Depth: {}".format(depth))

            depth_data = {
                t : {
                    "Bhattacharyya_Coefficient" : [],
                    "Bhattacharyya_Distance" : [],
                    "Hellinger_Distance" : [],
                    "JS_Divergence" : []
                } for t in trajectory_list
            }
            depth_loc = os.path.join(qubit_loc, f"Depth_{depth}")
            os.mkdir(depth_loc)

            circuit_loc = os.path.join(depth_loc, "Circuit_Instructions")
            os.mkdir(circuit_loc)

            trajectory_dirs = {
                t : os.path.join(depth_loc, f"{t}_Trajectories") for t in trajectory_list
            }

            for val in trajectory_dirs.values():
                os.mkdir(val)

            density_matrix_dir = os.path.join(depth_loc, "Density_Matrix")
            os.mkdir(density_matrix_dir)

            for trial in range(num_trials):
                
                save_location = os.path.join(circuit_loc, f"Circuit_Instructions_{trial}_Trial.json")
                generate_random_circuit(circuit, save_location, depth)

                p_ref = run_density_matrix_simulation(circuit)
                np.save(os.path.join(density_matrix_dir, f"Density_Matrix_Probabilities_{trial}_Trial.npy"), p_ref)

                for trajectories in trajectory_list:

                    p_comp = run_custom_simulation(circuit, trajectories)
                    np.save(os.path.join(trajectory_dirs[trajectories], f"MCWF_Probabilities_{trajectories}_Trajectories_{trial}_Trial.npy"), p_comp)

                    metrics = compute_fidelity_metrics(p_comp, p_ref)

                    for i,key in enumerate(depth_data[trajectories].keys()):
                        depth_data[trajectories][key].append(metrics[i])
                
            with open(os.path.join(depth_loc, f"Fidelity_Metrics_Depth_{depth}_Data.pkl"), "wb") as f:
                pickle.dump(depth_data, f)
            
            for trajectories in trajectory_list:
                data_logger.write(f"{qubit}\t|\t{depth}\t|\t{trajectories}\t|\t{np.mean(depth_data[trajectories]['Bhattacharyya_Coefficient']):.4f} ± {np.std(depth_data[trajectories]['Bhattacharyya_Coefficient']):.2E}\t|\t{np.mean(depth_data[trajectories]['Bhattacharyya_Distance']):.4f} ± {np.std(depth_data[trajectories]['Bhattacharyya_Distance']):.2E}\t|\t{np.mean(depth_data[trajectories]['Hellinger_Distance']):.4f} ± {np.std(depth_data[trajectories]['Hellinger_Distance']):.2E}\t|\t{np.mean(depth_data[trajectories]['JS_Divergence']):.4f} ± {np.std(depth_data[trajectories]['JS_Divergence']):.2E}\n")

            qubit_data[f"Depth_{depth}"] = depth_data
            data_logger.write("\n")
            data_logger.flush()

        data[f"{qubit}_Qubits"] = qubit_data
        data_logger.write("--"*20 + "\n\n")
        data_logger.flush()
        with open(os.path.join(qubit_loc, f"Fidelity_Metrics_Qubit_{qubit}_Data.pkl"), "wb") as f:
            pickle.dump(qubit_data, f)

    data_logger.close()

    with open(os.path.join(base_dir, "Fidelity_Benchmark_Consolidated_Data.pkl"), "wb") as f:
        pickle.dump(data, f)

    metric_name_list = ["Bhattacharyya_Coefficient", "Bhattacharyya_Distance", "Hellinger_Distance", "JS_Divergence"]
    metric_name_list_idx = 0
    plot_depth = 100
    plot(data, plot_depth, metric_name_list[metric_name_list_idx], os.path.join(base_dir, "Plot_for_{}_at_{}_Depth.svg".format(metric_name_list[metric_name_list_idx], plot_depth)))