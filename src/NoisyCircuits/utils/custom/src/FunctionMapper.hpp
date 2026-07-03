/**
 * This header file provides code to generate maps between the gate name and it's functions.
*/

#pragma once
#include "TypeDefs.hpp"
#include "QuantumGates.hpp"

/*
 * Function that maps the gate name to the function that applies the gate to the state
 * 
 * Inputs:
 *      None
 * 
 * Returns:
 *      std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, unsigned short)
 *          Map whose key is the name of the gate as a string and the return is the function call to apply the gate.
 */
static inline std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, uint8)> gate_function_mapper(){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, uint8)> gate_map;
    gate_map["x"] = apply_X_gate;
    gate_map["sx"] = apply_SX_gate;
    gate_map["rz"] = apply_RZ_gate;
    gate_map["rx"] = apply_RX_gate;
    gate_map["cz"] = apply_CZ_gate;
    gate_map["ecr"] = apply_ECR_gate;
    gate_map["rzz"] = apply_RZZ_gate;
    gate_map["unitary"] = apply_unitary_gate;
    gate_map["h"] = apply_H_gate;
    gate_map["cx"] = apply_CX_gate;
    gate_map["ry"] = apply_RY_gate;
    gate_map["p"] = apply_P_gate;
    gate_map["swap"] = apply_SWAP_gate;
    return gate_map;
}

/*
 * Function that maps the gate to the correct noise function applicator.
 * 
 * Inputs:
 *      None
 * 
 * Returns:
 *      std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std:size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, unsigned short)
 *          Map whose key is the gate name as a string and the return is the noise application function.
 */
static inline std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)> noise_function_mapper(){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, uint8)> apply_noise_map;
    std::list<std::string> single_qubit_gate_names = {
        "x",
        "sx",
        "rz",
        "rx",
        "ry",
        "p",
        "h"
    };
    std::list<std::string> two_qubit_gate_names = {
        "cz",
        "ecr",
        "rzz",
        "cx",
        "swap"
    };
    for (const std::string& gate_name : single_qubit_gate_names){
        apply_noise_map[gate_name] = apply_single_qubit_noise;
    }
    for (const std::string& gate_name : two_qubit_gate_names){
        apply_noise_map[gate_name] = apply_two_qubit_noise;
    }
    apply_noise_map["unitary"] = apply_noise_for_unitary_matrix;
    return apply_noise_map;
}