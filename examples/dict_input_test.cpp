#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <complex>
#include <list>
#include <iostream>
#include <unordered_map>


namespace py = pybind11;
using complex128 = std::complex<double>;
using matrix = std::vector<std::vector<complex128>>;
using noise_map = std::unordered_map<std::string, std::vector<matrix>>;

struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

using noise_map_2q = std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash>>;

static matrix to_matrix(const py::array& array_object){
    auto arr = py::array_t<complex128, py::array::c_style | py::array::forcecast>(array_object);
    auto a = arr.unchecked<2>();
    matrix noise_matrix(a.shape(0), std::vector<complex128>(a.shape(1)));
    for (unsigned short i = 0; i < a.shape(0); i++){
        #pragma GCC unroll 2
        for (unsigned short j = 0; j < a.shape(1); j++){
            noise_matrix[i][j] = a(i, j);
        }
    }
    return noise_matrix;
}

std::vector<noise_map> parse_single_qubit_noise(const py::dict& noise_instructions){
    std::vector<noise_map> noise_list;
    for (auto qubit_item : noise_instructions){
        int qubit = py::cast<int>(qubit_item.first);
        py::dict gate_dict = py::cast<py::dict>(qubit_item.second);
        noise_map qubit_noise_map;
        for (auto gate_item : gate_dict){
            std::string gate_name = py::cast<std::string>(gate_item.first);
            py::list matrix_list = py::cast<py::list>(gate_item.second);
            std::vector<matrix> noise_matrix_list;
            noise_matrix_list.reserve(py::len(matrix_list));
            for (auto matrix_item : matrix_list){
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            qubit_noise_map.emplace(gate_name, std::move(noise_matrix_list));
        }
        noise_list.push_back(std::move(qubit_noise_map));
    }
    return noise_list;
}

noise_map_2q parse_two_qubit_noise(const py::dict& noise_instructions){
    noise_map_2q noise_list;
    for (auto qubit_item : noise_instructions){
        std::string gate_name = py::cast<std::string>(qubit_item.first);
        py::dict qubit_pair_dict = py::cast<py::dict>(qubit_item.second);
        std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash> qubit_pair_map;
        for (auto qubit_pair_item : qubit_pair_dict){
            std::pair<int,int> qubit_pair = py::cast<std::pair<int,int>>(qubit_pair_item.first);
            py::list matrix_list = py::cast<py::list>(qubit_pair_item.second);
            std::vector<matrix> noise_matrix_list;
            noise_matrix_list.reserve(py::len(matrix_list));
            for (auto matrix_item : matrix_list){
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            qubit_pair_map.emplace(qubit_pair, std::move(noise_matrix_list));
        }
        noise_list.emplace(gate_name, std::move(qubit_pair_map));
    }
    return noise_list;
}

void run_main(py::dict noise_instructions, int qubit_number, std::string gate){
    std::vector<noise_map> noise_list = parse_single_qubit_noise(noise_instructions);
    if (qubit_number < noise_list.size() && noise_list[qubit_number].count(gate) > 0) {
        const auto& noise_operators = noise_list[qubit_number][gate];
        for (const auto& matrix : noise_operators) {
            for (const auto& row : matrix) {
                for (const auto& element : row) {
                    std::cout << element << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "----" << std::endl;
        }
    } else {
        std::cout << "No noise operators found for qubit " << qubit_number << " and gate " << gate << "." << std::endl;
    }
}

void run_main2q(py::dict noise_instructions, int qubit_1, int qubit_2, std::string gate){
    noise_map_2q noise_list = parse_two_qubit_noise(noise_instructions);
    std::pair<int,int> qubit_pair = {qubit_1, qubit_2};
    if (noise_list.count(gate) > 0 && noise_list[gate].count(qubit_pair) > 0) {
        const auto& noise_operators = noise_list[gate][qubit_pair];
        for (const auto& matrix : noise_operators) {
            for (const auto& row : matrix) {
                for (const auto& element : row) {
                    std::cout << element << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "----" << std::endl;
        }
    } else {
        std::cout << "No noise operators found for qubit pair (" << qubit_pair.first << ", " << qubit_pair.second << ") and gate " << gate << "." << std::endl;
    }
}

PYBIND11_MODULE(test_dictionary_processing, m){
    m.doc() = "Test Dictionary Processing";
    m.def("run_main", &run_main, "Process noise instructions.");
    m.def("run_main2q", &run_main2q, "Process two-qubit noise instructions.");
}