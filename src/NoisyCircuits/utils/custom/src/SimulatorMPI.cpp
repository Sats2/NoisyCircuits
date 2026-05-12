#include "TypeDefs.hpp"
#include "NoiseParser.hpp"
#include "QuantumGates.hpp"
#include "FunctionMapper.hpp"


void simulate_circuit_instance(complex128* __restrict__ state, const std::list<ItemEntry>& instruction_list, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t num_qubits, uint8 seed, uint8 thread_count){
    std::mt19937_64 trajectory_engine(seed);
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, uint8)> gate_map = gate_function_mapper();
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)> function_noise_map = noise_function_mapper();
    for (const ItemEntry& instruction : instruction_list){
        const std::string& gate_name = instruction.gate_name;
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const double params = instruction.params;
        const matrix& unitary_matrix = instruction.unitary_matrix;
        gate_map[gate_name](state, qubits[0], qubits[1], num_qubits, params, unitary_matrix, qubits, thread_count);
        std::vector<matrix> noise_matrix_list = get_matrix_list_for_instruction(gate_name, single_qubit_instructions, two_qubit_instructions, qubits[0], qubits[1]);
        function_noise_map[gate_name](state, qubits[0], qubits[1], num_qubits, noise_matrix_list, trajectory_engine, thread_count);
    }
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
        const complex128 amplitude = state[i];
        state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
    }
}

void run_trajectory(py::list instructions, py::array_t<complex128> statevector, py::dict single_qubit_noise_instructions, py::dict two_qubit_noise_instructions, std::size_t num_qubits, uint8 seed, uint8 thread_count){
    std::list<ItemEntry> instruction_list;
    for (auto item : instructions){
        auto item_tuple = item.cast<py::tuple>();
        ItemEntry entry;
        entry.gate_name = item_tuple[0].cast<std::string>();
        entry.qubits = item_tuple[1].cast<std::vector<std::size_t>>();
        if (entry.gate_name == "unitary"){
            py::array unitary_matrix_array = py::cast<py::array>(item_tuple[2]);
            auto arr = py::array_t<complex128, py::array::c_style | py::array::forcecast>(unitary_matrix_array);
            int dim = arr.ndim();
            auto a = arr.unchecked<-1>();
            matrix U(a.shape(0), std::vector<complex128>(a.shape(1)));
            for (std::size_t i = 0; i < dim; i++){
                #pragma GCC unroll 2
                for (std::size_t j = 0; j < dim; j++){
                    U[i][j] = a(i, j);
                }
            }
            entry.unitary_matrix = U;
            entry.params = -1.0;
        }
        else {
            entry.unitary_matrix = {};
            entry.params = item_tuple[2].cast<double>();
        }
        instruction_list.push_back(entry);
    }
    py::buffer_info state_info = statevector.request();
    auto * state = static_cast<complex128*>(state_info.ptr);
    std::vector<noise_map> single_qubit_instructions = parse_single_qubit_noise(single_qubit_noise_instructions);
    noise_map2q two_qubit_instructions = parse_two_qubit_noise(two_qubit_noise_instructions);
    simulate_circuit_instance(state, instruction_list, single_qubit_instructions, two_qubit_instructions, num_qubits, seed, thread_count);
}

PYBIND11_MODULE(simulator_mpi, m){
    m.doc() = "Module for simulating a single trajectory of the Monte-Carlo Wavefunction method acroos an entire node within an MPI environment.";
    m.def("run_trajectory", &run_trajectory, "Run a single trajectory of the noisy quantum circuit simulation across an entire node.", py::arg("instructions"), py::arg("statevector"), py::arg("single_qubit_instructions"), py::arg("two_qubit_instructions"), py::arg("num_qubits"), py::arg("seed"), py::arg("thread_count"));
}