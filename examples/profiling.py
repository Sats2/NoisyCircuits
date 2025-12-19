from NoisyCircuits import QuantumCircuit as QC
from NoisyCircuits.utils.GetNoiseModel import GetNoiseModel
import pickle
import os
from scipy.sparse import csr_matrix, csc_matrix
import json
from memory_profiler import profile

memory_log = open("profiling_memory.log", "w+")

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

@profile(stream=memory_log)
def memory_single_matrix(nqc):
    dense_mats = {}
    for q_num in nqc.single_qubit_instructions:
        q_res = {}
        for op_name in ["x", "sx"]:
            op_list = []
            ops = nqc.single_qubit_instructions[q_num][op_name]["kraus_operators"]
            for op in ops:
                op_list.append(op)
            q_res[op_name] = op_list
        dense_mats[q_num] = q_res
    return dense_mats    

@profile(stream=memory_log)
def memory_csr_matrix(nqc):
    sparse_mats = {}
    for q_num in nqc.single_qubit_instructions:
        q_res = {}
        for op_name in ["x", "sx"]:
            op_list = []
            ops = nqc.single_qubit_instructions[q_num][op_name]["kraus_operators"]
            for op in ops:
                op_list.append(csr_matrix(op))
            q_res[op_name] = op_list
        sparse_mats[q_num] = q_res
    return sparse_mats

@profile(stream=memory_log)
def memory_csc_matrix(nqc):
    sparse_mats = {}
    for q_num in nqc.single_qubit_instructions:
        q_res = {}
        for op_name in ["x", "sx"]:
            op_list = []
            ops = nqc.single_qubit_instructions[q_num][op_name]["kraus_operators"]
            for op in ops:
                op_list.append(csc_matrix(op))
            q_res[op_name] = op_list
        sparse_mats[q_num] = q_res
    return sparse_mats

@profile(stream=memory_log)
def dense_cz_mats(nqc):
    dense_mats = {}
    for key_pair in nqc.two_qubit_instructions["cz"].keys():
        ops = nqc.two_qubit_instructions["cz"][key_pair]["operators"]
        dense_mats[key_pair] = ops
    return dense_mats

@profile(stream=memory_log)
def csr_cz_mats(nqc):
    sparse_mats = {}
    for key_pair in nqc.two_qubit_instructions["cz"].keys():
        ops = nqc.two_qubit_instructions["cz"][key_pair]["operators"]
        op_list = []
        for op in ops:
            op_list.append(csr_matrix(op))
        sparse_mats[key_pair] = op_list
    return sparse_mats

@profile(stream=memory_log)
def csc_cz_mats(nqc):
    sparse_mats = {}
    for key_pair in nqc.two_qubit_instructions["cz"].keys():
        ops = nqc.two_qubit_instructions["cz"][key_pair]["operators"]
        op_list = []
        for op in ops:
            op_list.append(csc_matrix(op))
        sparse_mats[key_pair] = op_list
    return sparse_mats

if __name__ == "__main__":
    noise_model = get_noise_model()
    nqc = create_circuit_object(num_qubits=10,
                                noise_model=noise_model,
                                num_cores=4,
                                num_trajectories=1000,
                                threshold=1e-8,
                                jsonize=False,
                                verbose=False)
    single_qubit_dense = memory_single_matrix(nqc)
    single_qubit_csr = memory_csr_matrix(nqc)
    single_qubit_csc = memory_csc_matrix(nqc)
    two_qubit_dense = dense_cz_mats(nqc)
    two_qubit_csr = csr_cz_mats(nqc)
    two_qubit_csc = csc_cz_mats(nqc)
    nqc.shutdown()
    print("Completed profiling memory usage.")