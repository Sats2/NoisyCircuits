#include "TypeDefs.hpp"
#include "NoiseParser.hpp"
#include "QuantumGates.hpp"
#include "FunctionMapper.hpp"
#include <mpi.h>


std::vector<complex128> single_trajectory(const std::list<ItemEntry> instruction_list, const std::vector<noise_map> single_qubit_instructions, const noise_map2q two_qubit_instructions, const std::size_t num_qubits, uint8 seed){
    std::mt19937_64 trajectory_engine(seed);
    std::vector<complex128> trajectory_state(std::size_t{1} << num_qubits, 0.0);
    trajectory_state[0] = 1.0;
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, uint8)> gate_map = gate_function_mapper();
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)> noise_function_map = noise_function_mapper();
    for (const ItemEntry& instruction : instruction_list){
        const std::string& gate_name = instruction.gate_name;
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const double params = instruction.params;
        gate_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, params, 1);
        std::vector<matrix> noise_matrix_list = get_matrix_list_for_instruction(gate_name, single_qubit_instructions, two_qubit_instructions, qubits[0], qubits[1]);
        noise_function_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, noise_matrix_list, trajectory_engine, 1);
    }
    for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
        const complex128 amplitude = trajectory_state[i];
        trajectory_state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
    }
    return trajectory_state;
}