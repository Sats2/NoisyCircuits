from NoisyCircuits import QuantumCircuit as QC
from NoisyCircuits.utils.GetNoiseModel import GetNoiseModel
import pickle
import numpy as np
# from memory_read import PeakMemory
import tracemalloc
import pandas as pd

def get_noise_model():
    model = pickle.load(open("../noise_models/Noise_Model_Heron_QPU.pkl", "rb"))
    return model

def create_circuit_object(num_qubits,
                          noise_model,
                          num_cores,
                          num_trajectories,
                          threshold,
                          jsonize,
                          verbose):
    nqc = QC(num_qubits=num_qubits, 
         noise_model=noise_model, 
         num_cores=num_cores,
         backend_qpu_type="heron", 
         num_trajectories=num_trajectories, 
         threshold=threshold, 
         jsonize=jsonize,
         verbose=verbose)
    return nqc

def build_random_circuit(circuit, depth, num_qubits):
    circuit.refresh()
    gate = {
        "X" : circuit.X,
        "SX": circuit.SX,
    }
    for _ in range(depth):
        for i in range(num_qubits):
            choice = np.random.choice(["X", "SX"])
            gate[choice](i)
    for i in range(num_qubits-1):
        circuit.CZ(i, i+1)

def execute_circuit(circuit, num_qubits, method, trajectories):
    if method == "dm":
        p = circuit.run_with_density_matrix(list(range(num_qubits)))
    else:
        p = circuit.execute(list(range(num_qubits)), num_trajectories=trajectories)
    return p

if __name__ == "__main__":
    noise_model = get_noise_model()
    data = pd.DataFrame(columns=["trial_number", "num_qubits", "depth", "peak_memory_bytes", "current_memory_bytes"])
    for qubits in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        circuit = QC(num_qubits=qubits, 
                    noise_model=noise_model,
                    backend_qpu_type="heron", 
                    num_trajectories=1, 
                    num_cores=1, 
                    threshold=1e-8, 
                    jsonize=True,
                    verbose=False)
        for depth in [1, 5, 20, 50, 100, 250]:
            build_random_circuit(circuit, depth=depth, num_qubits=qubits)
            for trial in [1, 2, 3, 4, 5]:
                tracemalloc.start()
                res = execute_circuit(circuit, qubits, method="density_matrix", trajectories=100)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                del res
                data.loc[len(data)] = {"trial_number":trial,
                                        "num_qubits": qubits,
                                        "depth": depth,
                                        "peak_memory_bytes": peak,
                                        "current_memory_bytes": current}
            print("Completed qubits:", qubits, " depth:", depth)
            data.to_csv("intermediate/profiling_after_{}q_{}d.csv".format(qubits, depth), index=False)
        circuit.shutdown()
    data.to_csv("profiling_results_density_matrix.csv", index=False)