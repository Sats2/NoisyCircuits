#pragma once
#include "TypeDefs.hpp"
#include "QuantumGates.hpp"


static inline std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, uint8)> gate_function_mapper(){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, uint8)> gate_map;
    gate_map["x"] = apply_X_gate;
    gate_map["sx"] = apply_SX_gate;
    gate_map["rz"] = apply_RZ_gate;
    gate_map["rx"] = apply_RX_gate;
    gate_map["cz"] = apply_CZ_gate;
    gate_map["ecr"] = apply_ECR_gate;
    gate_map["rzz"] = apply_RZZ_gate;
    return gate_map;
}

static inline std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)> noise_function_mapper(){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)> apply_noise_map;
    std::list<std::string> single_qubit_gate_names = {
        "x",
        "sx",
        "rz",
        "rx"
    };
    std::list<std::string> two_qubit_gate_names = {
        "cz",
        "ecr",
        "rzz"
    };
    for (const std::string& gate_name : single_qubit_gate_names){
        apply_noise_map[gate_name] = apply_single_qubit_noise;
    }
    for (const std::string& gate_name : two_qubit_gate_names){
        apply_noise_map[gate_name] = apply_two_qubit_noise;
    }
    return apply_noise_map;
}