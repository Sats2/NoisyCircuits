#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <complex>
#include <list>
#include <unordered_map>
#include <random>
#include <string>
#include <omp.h>
#include "SimulatorGateImplementation.cpp"

namespace py = pybind11;

struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct ItemEntry {
    std::string gate_name;
    std::vector<std::size_t> qubits;
    double params;
};

using complex128 = std::complex<double>;
using uint8 = const unsigned short;
using matrix = std::vector<std::vector<complex128>>;
using noise_map = std::unordered_map<std::string, std::vector<matrix>>;
using noise_map2q = std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash>>;

void set_thread_count(uint8 thread_count){
    omp_set_num_threads(thread_count);
}

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

static std::vector<noise_map> parse_single_qubit_noise(const py::dict& noise_dictionary){
    std::vector<noise_map> noise_list;
    for (auto item : noise_dictionary){
        int qubit = py::cast<int>(item.first);
        py::dict gate_noise_dict = py::cast<py::dict>(item.second);
        noise_map gate_noise_map;
        for (auto gate_item : gate_noise_dict){
            std::string gate_name = py::cast<std::string>(gate_item.first);
            py::list matrix_list = py::cast<py::list>(gate_item.second);
            std::vector<matrix> noise_matrix_list;
            noise_matrix_list.reserve(py::len(matrix_list));
            for (auto matrix_item : matrix_list){
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            gate_noise_map.emplace(gate_name, std::move(noise_matrix_list));
        }
        noise_list.push_back(std::move(gate_noise_map));
    }
    return noise_list;
}

static noise_map2q parse_two_qubit_noise(const py::dict& noise_dictionary){
    noise_map2q noise_list;
    for (auto item : noise_dictionary){
        std::string gate_name = py::cast<std::string>(item.first);
        py::dict qubit_pair_dict = py::cast<py::dict>(item.second);
        std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash> qubit_pair_map;
        for (auto pair_item : qubit_pair_dict){
            std::pair<int, int> qubit_pair = py::cast<std::pair<int, int>>(pair_item.first);
            py::list matrix_list = py::cast<py::list>(pair_item.second);
            std::vector<matrix> noise_matrix_list;
            noise_matrix_list.reserve(py::len(matrix_list));
            for (auto matrix_item : matrix_list) {
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            qubit_pair_map.emplace(qubit_pair, std::move(noise_matrix_list));
        }
        noise_list.emplace(gate_name, std::move(qubit_pair_map));
    }
    return noise_list;
}