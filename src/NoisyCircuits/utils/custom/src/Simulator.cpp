#include "TypeDefs.hpp"
#include "NoiseParser.hpp"
#include "QuantumGates.hpp"
#include "FunctionMapper.hpp"


void pure_state_solver(const std::list<ItemEntry> instruction_list, std::size_t num_qubits, complex128* __restrict__ state, bool return_state, uint8 thread_count){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, uint8)> gate_map = gate_function_mapper();
    state[0] = complex128(1.0, 0.0);
    for (const ItemEntry& instruction : instruction_list){
        const std::string& gate_name = instruction.gate_name;
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const double params = instruction.params;
        const matrix& unitary_matrix = instruction.unitary_matrix;
        gate_map[gate_name](state, qubits[0], qubits[1], num_qubits, params, unitary_matrix, qubits, thread_count);
    }
    if (!return_state) {
        #pragma omp parallel for num_threads(thread_count)
        for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
            const complex128 amplitude = state[i];
            state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
        }
    }
}

std::vector<complex128> single_trajectory(const std::list<ItemEntry>& instruction_list, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t num_qubits, std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, uint8)>& gate_map, std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)>& noise_function_map, uint8 seed){
    std::mt19937_64 trajectory_engine(seed);
    std::vector<complex128> trajectory_state(std::size_t{1} << num_qubits, 0.0);
    trajectory_state[0] = 1.0;
    for (const ItemEntry& instruction : instruction_list){
        const std::string& gate_name = instruction.gate_name;
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const double params = instruction.params;
        const matrix& unitary_matrix = instruction.unitary_matrix;
        gate_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, params, unitary_matrix, qubits, 1);
        std::vector<matrix> noise_matrix_list = get_matrix_list_for_instruction(gate_name, single_qubit_instructions, two_qubit_instructions, qubits[0], qubits[1]);
        noise_function_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, noise_matrix_list, trajectory_engine, 1);
    }
    for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
        const complex128 amplitude = trajectory_state[i];
        trajectory_state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
    }
    return trajectory_state;
}

void monte_carlo_wavefunction_solver(const std::list<ItemEntry> instruction_list, complex128* __restrict__ state, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t num_qubits, const int num_trajectories, uint8 thread_count){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, uint8)> gate_map = gate_function_mapper();
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)> noise_function_map = noise_function_mapper();
    double averaging_factor = 1 / static_cast<double>(num_trajectories);
    #pragma omp parallel for shared(instruction_list, num_qubits, single_qubit_instructions, two_qubit_instructions, gate_map, noise_function_map) schedule(dynamic) num_threads(thread_count)
    for (unsigned short t = 0; t < num_trajectories; t++){
        uint8 seed = static_cast<uint8>(42 + t);
        std::vector<complex128> trajectory_state = single_trajectory(instruction_list, single_qubit_instructions, two_qubit_instructions, num_qubits, gate_map, noise_function_map, seed);
        #pragma omp critical
        {
            for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
                state[i] += (trajectory_state[i] * averaging_factor);
            }
        }
    }
}

void simulate_circuit(py::list instructions, py::array_t<complex128> statevector, py::dict single_qubit_noise_instructions, py::dict two_qubit_noise_instructions, std::size_t num_qubits, bool noisy, uint8 num_trajectories, bool return_state, uint8 thread_count){
    std::list<ItemEntry> instruction_list;
    for (auto item : instructions){
        auto item_tuple = item.cast<py::tuple>();
        ItemEntry entry;
        entry.gate_name = item_tuple[0].cast<std::string>();
        std::vector<std::size_t> apply_to_qubits = item_tuple[1].cast<std::vector<std::size_t>>();
        if (entry.gate_name == "unitary"){
            py::array unitary_matrix_array = py::cast<py::array>(item_tuple[2]);
            auto arr = py::array_t<complex128, py::array::c_style | py::array::forcecast>(unitary_matrix_array);
            auto a = arr.unchecked<-1>();
            matrix U(a.shape(0), std::vector<complex128>(a.shape(1)));
            std::size_t dim = a.shape(0);
            for (std::size_t i = 0; i < dim; i++){
                #pragma GCC unroll 2
                for (std::size_t j = 0; j < dim; j++){
                    U[i][j] = a(i, j);
                }
            }
            entry.unitary_matrix = U;
            entry.params = -1.0;
            std::reverse(apply_to_qubits.begin(), apply_to_qubits.end());
            entry.qubits = apply_to_qubits;
        }
        else {
            entry.unitary_matrix = {};
            entry.qubits = apply_to_qubits;
            entry.params = item_tuple[2].cast<double>();
        }
        instruction_list.push_back(entry);
    }
    py::buffer_info state_info = statevector.request();
    auto * state = static_cast<complex128*>(state_info.ptr);
    if (noisy){
        std::vector<noise_map> single_qubit_instructions = parse_single_qubit_noise(single_qubit_noise_instructions);
        noise_map2q two_qubit_instructions = parse_two_qubit_noise(two_qubit_noise_instructions);
        monte_carlo_wavefunction_solver(instruction_list, state, single_qubit_instructions, two_qubit_instructions, num_qubits, num_trajectories, thread_count);
    }
    else {
        unsigned short max_usable_threads = std::min<unsigned short>(thread_count, num_qubits);
        max_usable_threads = std::min<unsigned short>(max_usable_threads, omp_get_max_threads());
        pure_state_solver(instruction_list, num_qubits, state, return_state, max_usable_threads);
    }
}

PYBIND11_MODULE(simulator, m){
    m.doc() = "Module for simulating quantum circuits with noise using the Monte Carlo Wavefunction method.";
    m.def("simulate_circuit", &simulate_circuit, "Simulates a quantum circuit with and without noise.", py::arg("instructions"), py::arg("statevector"), py::arg("single_qubit_noise_instructions"), py::arg("two_qubit_noise_instructions"), py::arg("num_qubits"), py::arg("noisy"), py::arg("num_trajectories"), py::arg("return_state"), py::arg("thread_count"));
}