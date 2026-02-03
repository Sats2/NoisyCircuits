from NoisyCircuits.utils.BuildQubitGateModelParallel import BuildModel
import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("../noise_models/Noise_Model_Heron_QPU.pkl", "rb") as f:
    noise_model = pickle.load(f)

data = {
    "qubits" : [],
    "single_qubit_extraction_time" : [[], []],
    "two_qubit_extraction_time" : [[], []],
    "measurement_extraction_time" : [[], []],   
    "single_qubit_error_postprocessing_time": [[], []],
    "two_qubit_error_postprocessing_time": [[], []],
    "two_qubit_error_operator_construction_time": [[], []],
    "connectivity_map_construction_time": [[], []],
}
raw_data = {}

for qubits in range(2, 21):
    trial_data = {
        "single_qubit_extraction_time" : [],
        "two_qubit_extraction_time" : [],
        "measurement_extraction_time" : [],   
        "single_qubit_error_postprocessing_time": [],
        "two_qubit_error_postprocessing_time": [],
        "two_qubit_error_operator_construction_time": [],
        "connectivity_map_construction_time": [],
    }
    for _ in range(5):
        vals = BuildModel(
            noise_model=noise_model,
            num_qubits=qubits,
            num_cores=2,
            threshold=1e-15,
            basis_gates=[["sx", "x", "rz", "rx"], ["cz", "rzz"]],
            verbose=False).build_qubit_gate_model()
        for key in vals[-1]:
            trial_data[key].append(vals[-1][key])
        raw_data.setdefault(qubits, []).append(vals[-1])
    data["qubits"].append(qubits)
    for key in trial_data:
        data[key][0].append(np.mean(trial_data[key]))
        data[key][1].append(np.std(trial_data[key]))

with open("noise_build_times.pkl", "wb") as f:
    pickle.dump(data, f)

key_names = list(data.keys())
for key in key_names:
    if key == "qubits":
        continue
    plt.semilogy(data["qubits"], np.array(data[key][0])/1e9, label=key)
    plt.fill_between(data["qubits"], 
                     np.array(data[key][0])/1e9 - 2*np.array(data[key][1])/1e9,
                     np.array(data[key][0])/1e9 + 2*np.array(data[key][1])/1e9, alpha=0.2)
plt.xlabel("Number of Qubits")
plt.ylabel("Time (s)")
plt.grid()
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2)
plt.savefig("noise_build_times.png", bbox_inches='tight', dpi=300)
plt.show()