#pragma once
#include "TypeDefs.hpp"

static inline matrix to_matrix(const py::array& array_object){
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
};

static inline std::vector<noise_map> parse_single_qubit_noise(const py::dict& noise_dictionary){
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

static inline noise_map2q parse_two_qubit_noise(const py::dict& noise_dictionary){
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
            for (auto matrix_item : matrix_list){
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            qubit_pair_map.emplace(qubit_pair, std::move(noise_matrix_list));
        }
        noise_list.emplace(gate_name, std::move(qubit_pair_map));
    }
    return noise_list;
}

std::vector<matrix> get_matrix_list_for_instruction(const std::string& gate_name, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t q1, const std::size_t q2){
    if (q1 == q2){
        std::vector<matrix> noise_matrix_list = single_qubit_instructions[q1].at(gate_name);
        return noise_matrix_list;
    }
    else {
        std::pair<int, int> key = {q1, q2};
        std::vector<matrix> noise_matrix_list = two_qubit_instructions.at(gate_name).at(key);
        return noise_matrix_list;
    }
}