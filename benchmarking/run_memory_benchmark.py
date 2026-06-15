import subprocess
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

trials = 35
max_qubits = 16
trajectory_list = [10, 100, 500, 1000]
file_location = os.path.join(os.path.expanduser("~"), "benchmarking_data/Memory/")

data = {
    "initialization" : {},
    "mcwf" : {},
    "density_matrix" : {}
}

def benchmark_memory(qubits, trajectories, run_density_matrix, trial, file_loc, circuit_loc):
    if run_density_matrix:
        file = open(file_loc + "/Memory_Output_{}_Q_density_matrix_{}_Trial.txt".format(qubits, trial), "w")
        intro_line = "Trial No. {} for Density Matrix Simulations.\nMemory usage for {} Qubits.\n\n".format(trial, qubits, trajectories)
    else:
        file = open(file_loc + "/Memory_Output_{}_Q_mcwf_{}_Trajectories_{}_Trial.txt".format(qubits, trajectories, trial), "w")
        intro_line = "Trial No. {} for MCWF Simulation.\nMemory usage for {} Qubits, {} Trajectories.\n\n".format(trial, qubits, trajectories)
    
    output = subprocess.run(["/usr/bin/time", "-v", "python3", "method_execution_script.py", str(qubits), str(circuit_loc), str(run_density_matrix), str(trajectories)], capture_output=True, text=True)

    out = output.stderr
    file.write(intro_line)
    file.write(out)
    file.close()
    match = re.search("Maximum resident set size \(kbytes\):\s+(\d+)", out)
    if match:
        mem = int(match.group(1))
        return mem / 1024
    return None

def benchmark_memory_initialization(qubits, depth, save_loc, trial, file_loc):
    file = open(file_loc + "/Memory_Output_{}_Q_Initialization_Trial_{}.txt".format(qubits, trial), "w")
    intro_line = "Trial No. {} for Circuit Initialization.\nMemory usage for {} Qubits, Depth {}.\n\n".format(trial, qubits, depth)
    output = subprocess.run(["/usr/bin/time", "-v", "python3", "initialization_script.py", str(qubits), str(depth), str(save_loc)], capture_output=True, text=True)
    out = output.stderr
    file.write(intro_line)
    file.write(out)
    file.close()
    match = re.search("Maximum resident set size \(kbytes\):\s+(\d+)", out)
    if match:
        mem = int(match.group(1))
        return mem / 1024
    return None



consolidated_output = open(file_location + "/Memory_Benchmark_Consolidated_Output.txt", "w")
for qubits in range(2, max_qubits + 1):
    text = "Qubits: {}\n".format(qubits)
    qubit_data = {}
    file_loc_qubit = file_location + "{}_Qubits".format(qubits)
    os.mkdir(file_loc_qubit)
    
    mem_usage_init_list = []
    for t in trajectory_list:
        qubit_data[t] = {
            "total_usage" : [],
            "process_usage" : []
        }
    density_matrix_memory_usages = []
    density_matrix_memory_usages_full = []
    os.mkdir(file_loc_qubit + "/Circuit_Instructions")

    for trial in range(trials):
        circuit_loc = file_loc_qubit + "/Circuit_Instructions/Circuit_Object_{}Q_{}Trial.json".format(qubits, trial)
        mem_usage_init = benchmark_memory_initialization(qubits, 100, circuit_loc, trial, file_loc_qubit)
        mem_usage_init_list.append(mem_usage_init)
        for trajectory_number in trajectory_list:
            file_loc_traj = file_loc_qubit + "/{}_Trajectories".format(trajectory_number)
            if not os.path.exists(file_loc_traj):
                os.mkdir(file_loc_traj)
            memory_usage = benchmark_memory(qubits, trajectory_number, False, trial, file_loc_traj, circuit_loc)
            qubit_data[trajectory_number]["total_usage"].append(memory_usage)
            qubit_data[trajectory_number]["process_usage"].append(memory_usage - mem_usage_init)
        file_loc_dm = file_loc_qubit + "/Density_Matrix"
        if not os.path.exists(file_loc_dm):
            os.mkdir(file_loc_dm)
        memory_usage = benchmark_memory(qubits, trajectory_number, True, trial, file_loc_dm, circuit_loc)
        density_matrix_memory_usages.append(memory_usage - mem_usage_init)
        density_matrix_memory_usages_full.append(memory_usage)

    data["mcwf"][qubits] = qubit_data
    data["density_matrix"][qubits] = {
        "total_usage" : density_matrix_memory_usages_full,
        "process_usage" : density_matrix_memory_usages
    }
    data["initialization"][qubits] = mem_usage_init_list
    for trajectory_number in trajectory_list:
        mean_total_usage = np.mean(qubit_data[trajectory_number]["total_usage"])
        std_total_usage = np.std(qubit_data[trajectory_number]["total_usage"])
        mean_process_usage = np.mean(qubit_data[trajectory_number]["process_usage"])
        std_process_usage = np.std(qubit_data[trajectory_number]["process_usage"])
        text += "Trajectory Number: {}\n".format(trajectory_number)
        text += "Mean Total Memory Usage: {:.2f} MB, Std Dev: {:.2f} MB\n".format(mean_total_usage, std_total_usage)
        text += "Mean Process Memory Usage: {:.2f} MB, Std Dev: {:.2f} MB\n\n".format(mean_process_usage, std_process_usage)
    mean_dm_usage = np.mean(density_matrix_memory_usages)
    std_dm_usage = np.std(density_matrix_memory_usages)
    mean_dm_usage_full = np.mean(density_matrix_memory_usages_full)
    std_dm_usage_full = np.std(density_matrix_memory_usages_full)
    text += "Density Matrix Simulation:\n"
    text += "Mean Total Memory Usage: {:.2f} MB, Std Dev: {:.2f} MB\n".format(mean_dm_usage_full, std_dm_usage_full)
    text += "Mean Memory Usage: {:.2f} MB, Std Dev: {:.2f} MB\n".format(mean_dm_usage, std_dm_usage)
    text += "----------------------------------------\n\n"
    consolidated_output.write(text)
    consolidated_output.flush()
consolidated_output.close()


with open(file_location + "/Memory_Benchmark_Data.pkl", "wb") as f:
    pickle.dump(data, f)

def plot_data(data, file_loc, traj_list=trajectory_list):
    fig = plt.figure(figsize=(12,6))
    qubit_list = data["mcwf"].keys()
    mean_mem_usage_dm = []
    std_mem_usage_dm = []
    for qubits in qubit_list:
        mean_mem_usage_dm.append(np.mean(data["density_matrix"][qubits]["process_usage"]))
        std_mem_usage_dm.append(np.std(data["density_matrix"][qubits]["process_usage"]))
    plt.plot(qubit_list, mean_mem_usage_dm, label="Density Matrix")
    plt.fill_between(
        qubit_list,
        np.array(mean_mem_usage_dm) - 2*np.array(std_mem_usage_dm),
        np.array(mean_mem_usage_dm) + 2*np.array(std_mem_usage_dm),
        alpha=0.2
    )
    for traj in traj_list:
        mean_mem_usage_mcwf = []
        std_mem_usage_mcwf = []
        for qubits in qubit_list:
            mean_mem_usage_mcwf.append(np.mean(data["mcwf"][qubits][traj]["process_usage"]))
            std_mem_usage_mcwf.append(np.std(data["mcwf"][qubits][traj]["process_usage"]))
        plt.plot(qubit_list, mean_mem_usage_mcwf, label="MCWF - {} Trajectories".format(traj))
        plt.fill_between(
            qubit_list,
            np.array(mean_mem_usage_mcwf) - 2*np.array(std_mem_usage_mcwf),
            np.array(mean_mem_usage_mcwf) + 2*np.array(std_mem_usage_mcwf),
            alpha=0.2
        )
    plt.xlabel("Number of Qubits")
    plt.ylabel("Maximum Memory Usage (MB)")
    plt.title("Memory Usage of Density Matrix vs MCWF Simulations (Process Memory Usage)")
    plt.legend()
    plt.grid()
    plt.savefig(file_loc + "/Memory_Benchmark.svg")
    plt.clf()

plot_data(data, file_location)