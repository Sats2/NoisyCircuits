from NoisyCircuits import QuantumCircuit as QC
from NoisyCircuits.utils.GetNoiseModel import GetNoiseModel
import pickle
import os
import json
from memory_profiler import profile

memory_log = open("profiling_memory_with_sparse_multicore_newversion.log", "w")
memory_log.write("Memory profiling log for NoisyCircuits with 10 cores parallelization\n")
memory_log.flush()

@profile(stream=memory_log)
def get_noise_model():
    model = pickle.load(open("../noise_models/Noise_Model_Heron_QPU.pkl", "rb"))
    return model

@profile(stream=memory_log)
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


if __name__ == "__main__":
    noise_model = get_noise_model()
    for num_qubits in range(2, 11):
        memory_log.write(f"\nProfiling for {num_qubits} qubits:\n")
        memory_log.flush()
        nqc = create_circuit_object(num_qubits=num_qubits,
                                    noise_model=noise_model,
                                    num_cores=10,
                                    num_trajectories=50,
                                    threshold=1e-8,
                                    jsonize=False,
                                    verbose=False)
        nqc.shutdown()
        memory_log.write(f"Completed profiling for {num_qubits} qubits.\n")
        memory_log.flush()
    memory_log.close()
    print("Completed profiling memory usage.")