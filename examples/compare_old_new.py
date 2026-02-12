from NoisyCircuits.utils.BuildQubitGateModelParallel import BuildModel as Parallel
from NoisyCircuits.utils.BuildQubitGateModelSingleOptimized import BuildModel as SingleOpt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import identity, csr_matrix


with open("../noise_models/Noise_Model_Heron_QPU.pkl", "rb") as f:
    noise_model = pickle.load(f)

def compare(n_qubits):

    def compare_sparse(mat1, mat2):
        mat1 = csr_matrix(mat1).toarray()
        mat2 = csr_matrix(mat2).toarray()
        return np.allclose(mat1, mat2)
    
    def compare_dictionaries(dict1, dict2):
        single_old, two_old = dict1
        single_new, two_new = dict2
        for qubit in single_old.keys():
            for gate in single_old[qubit].keys():
                kraus_ops_new = single_new[qubit][gate]["kraus_operators"]
                kraus_ops_old = single_old[qubit][gate]["kraus_operators"]
                for k in range(len(kraus_ops_new)):
                    assert compare_sparse(kraus_ops_new[k], kraus_ops_old[k]), f"Single qubit gate mismatch on qubit {qubit} gate {gate} operator {k}"
                qubit_channel_new = single_new[qubit][gate]["qubit_channel"]
                qubit_channel_old = single_old[qubit][gate]["qubit_channel"]
                for k in range(len(qubit_channel_new)):
                    assert compare_sparse(qubit_channel_new[k], qubit_channel_old[k]), f"Single qubit channel mismatch on qubit {qubit} gate {gate} operator {k}"
            
        for gate in two_old.keys():
            for qubit_pair in two_old[gate].keys():
                kraus_ops_new = two_new[gate][qubit_pair]["operators"]
                kraus_ops_old = two_old[gate][qubit_pair]["operators"]
                for k in range(len(kraus_ops_new)):
                    assert compare_sparse(kraus_ops_new[k], kraus_ops_old[k]), f"Two qubit gate mismatch on qubits {qubit_pair} gate {gate} operator {k}"
                gate_channel_new = two_new[gate][qubit_pair]["qubit_channel"]
                gate_channel_old = two_old[gate][qubit_pair]["qubit_channel"]
                for k in range(len(gate_channel_new)):
                    assert compare_sparse(gate_channel_new[k], gate_channel_old[k]), f"Two qubit channel mismatch on qubits {qubit_pair} gate {gate} operator {k}"
        return True
    
    time_data = {
        # "Old" : [],
        "Single": [],
        "Parallel": []
    }
    for _ in range(3):
        # t0 = time.perf_counter_ns()
        # vals_old = Single(
        #             noise_model=noise_model,
        #             num_qubits=n_qubits,
        #             threshold=1e-15,
        #             basis_gates=[["sx", "x", "rz", "rx"], ["cz", "rzz"]],
        #             verbose=False).build_qubit_gate_model()
        # t1 = time.perf_counter_ns()
        # time_data["Old"].append(t1 - t0)
        t0 = time.perf_counter_ns()
        vals_single = SingleOpt(
                    noise_model=noise_model,
                    num_qubits=n_qubits,
                    num_cores=n_qubits,
                    threshold=1e-15,
                    basis_gates=[["sx", "x", "rz", "rx"], ["cz", "rzz"]],
                    verbose=False).build_qubit_gate_model()
        t1 = time.perf_counter_ns()
        time_data["Single"].append(t1 - t0)
        t0 = time.perf_counter_ns()
        vals_parallel = Parallel(
                    noise_model=noise_model,
                    num_qubits=n_qubits,
                    num_cores=n_qubits,
                    threshold=1e-15,
                    basis_gates=[["sx", "x", "rz", "rx"], ["cz", "rzz"]],
                    verbose=False).build_qubit_gate_model()
        t1 = time.perf_counter_ns()
        time_data["Parallel"].append(t1 - t0)
    # single_old, two_old, _, _ = vals_old
    single_single, two_single, _, _ = vals_single
    single_parallel, two_parallel, _, _ = vals_parallel
    # if compare_dictionaries((single_old, two_old), (single_single, two_single)) and compare_dictionaries((single_old, two_old), (single_parallel, two_parallel)):
    #     return time_data
    if compare_dictionaries((single_single, two_single), (single_parallel, two_parallel)):
        return time_data
    raise ValueError("Models do not match.")

if __name__ == "__main__":
    qubits_data = {
        "qubits":[],
        "mean": [[], []],
        "std": [[], []]
    }
    for n_qubits in range(1, 16):
        data = compare(n_qubits)
        qubits_data["qubits"].append(n_qubits)
        for i, key in enumerate(["Single", "Parallel"]):
            qubits_data["mean"][i].append(np.mean(data[key]) / 1e9)
            qubits_data["std"][i].append(np.std(data[key]) / 1e9)
    plt.figure(figsize=(10, 6))
    label_list = ["Single Core Optimized", "Parallel Implementation"]
    for i in range(2):
        plt.plot(qubits_data["qubits"], qubits_data["mean"][i], label=label_list[i])
        plt.fill_between(qubits_data["qubits"],
                        np.array(qubits_data["mean"][i]) - np.array(qubits_data["std"][i]),
                        np.array(qubits_data["mean"][i]) + np.array(qubits_data["std"][i]),
                        alpha=0.2)
    plt.xlabel("Number of Qubits")
    plt.ylabel("Time to Build Model (s)")
    plt.title("Comparison of Qubit Gate Model Building Times")
    plt.legend()
    plt.grid()
    plt.savefig("Single_vs_Parallel_Build_Times.png", dpi=300)
    plt.clf()
    with open("build_times.pkl", "wb") as f:
        pickle.dump(qubits_data, f)