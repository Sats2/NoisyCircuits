from NoisyCircuits.utils.BuildQubitGateModelParallel import BuildModel as New
from NoisyCircuits.utils.BuildQubitGateModel import BuildModel as Old
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import identity, csr_matrix


with open("../noise_models/Noise_Model_Heron_QPU.pkl", "rb") as f:
    noise_model = pickle.load(f)

def compare(single_old, single_new, two_old, two_new):
    def compare_sparse(mat1, mat2):
        mat1 = csr_matrix(mat1).toarray()
        mat2 = csr_matrix(mat2).toarray()
        return np.allclose(mat1, mat2)

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
    return 0

def compare_runtime(max_qubits):
    max_qubits += 1
    timer_old = {"mean": [], "std":[]}
    timer_new = {"mean": [], "std":[]}
    qubits_list = list(range(1, max_qubits))
    for qubits in range(1, max_qubits):
        trials_old = []
        trials_new = []
        for _ in range(3):
            t0 = time.perf_counter_ns()
            v_old = Old(noise_model, qubits, 1e-15, [["sx", "x", "rz", "rx"], ["cz", "rzz"]], False).build_qubit_gate_model()
            t1 = time.perf_counter_ns()
            trials_old.append(t1 - t0)
            t0 = time.perf_counter_ns()
            v_new = New(noise_model, qubits, 2, 1e-15, [["sx", "x", "rz", "rx"], ["cz", "rzz"]], False).build_qubit_gate_model()
            t1 = time.perf_counter_ns()
            trials_new.append(t1 - t0)
        single_old, two_old, _, _ = v_old
        single_new, two_new, _, _ = v_new
        compare(single_old, single_new, two_old, two_new)
        timer_old["mean"].append(np.mean(trials_old))
        timer_old["std"].append(np.std(trials_old))
        timer_new["mean"].append(np.mean(trials_new))
        timer_new["std"].append(np.std(trials_new))
        with open(f"temp/time_comparison_{qubits}_qubits.pkl", "wb") as f:
            pickle.dump({"old":timer_old, "new":timer_new}, f)
        print(f"Completed Timing Comparison for {qubits} qubits.")
    plt.figure(figsize=(10,6))
    plt.plot(qubits_list, np.array(timer_old["mean"])/1e9, label="Old Implementation", color="blue")
    plt.fill_between(qubits_list, (np.array(timer_old["mean"])-np.array(timer_old["std"]))/1e9, (np.array(timer_old["mean"])+np.array(timer_old["std"]))/1e9, color="blue", alpha=0.2)
    plt.plot(qubits_list, np.array(timer_new["mean"])/1e9, label="New Implementation", color="orange")
    plt.fill_between(qubits_list, (np.array(timer_new["mean"])-np.array(timer_new["std"]))/1e9, (np.array(timer_new["mean"])+np.array(timer_new["std"]))/1e9, color="orange", alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Time to Build Model (s)")
    plt.title("Comparison of Old and New Qubit Gate Model Building Implementations")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig("Runtime_Comparison.png", dpi=300)
    plt.show()
    time_data = {
        "old": timer_old,
        "new": timer_new
    }
    return time_data

if __name__ == "__main__":
    time_data = compare_runtime(15)
    with open("Runtime_Comparison_Data.pkl", "wb") as f:
        pickle.dump(time_data, f)