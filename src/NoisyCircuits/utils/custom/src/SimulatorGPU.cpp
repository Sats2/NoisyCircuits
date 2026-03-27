/*
This code is part of NoisyCircuits, licensed under the MIT License (https://opensource.org/licenses/MIT).
*/

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

std::mt19937 global_engine;

void set_seed(int seed){
    global_engine.seed(seed);
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
            noise_matrix_list.reserve(matrix_list.size());
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
            for (auto matrix_item : matrix_list){
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            qubit_pair_map.emplace(qubit_pair, std::move(noise_matrix_list));
        }
        noise_list.emplace(gate_name, std::move(qubit_pair_map));
    }
    return noise_list;
}

static inline double compute_probability(const complex128* __restrict__ state, const std::size_t dim){
    double probability = 0.0;
    #pragma omp parallel for reduction(+:probability)
    for (std::size_t i = 0; i < dim; i++){
        probability += state[i].real() * state[i].real()  + state[i].imag() * state[i].imag();
    }
    return probability;
}

static inline std::vector<complex128> apply_single_qubit_noise_operator(const complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const std::size_t dim, const std::size_t stride){
    std::vector<complex128> new_state = std::vector<complex128>(dim, complex128(0.0, 0.0));
    #pragma omp parallel for
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        new_state[i] += u00 * s0 + u01 * s1;
        new_state[j] += u10 * s0 + u11 * s1;
    }
    return new_state;
}

static inline void apply_inplace_operator_1q(complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const std::size_t dim, const std::size_t stride){
    #pragma omp parallel for
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = u00 * s0 + u01 * s1;
        state[j] = u10 * s0 + u11 * s1;
    }
}

static inline void apply_single_qubit_noise(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const std::vector<matrix>& noise_operators){
    const int num_operators = noise_operators.size();
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    std::size_t i,j;
    int counter = 0;
    std::vector<double> probability_list(num_operators, 0.0);
    for (const matrix& oper : noise_operators){
        complex128 u00 = oper[0][0];
        complex128 u01 = oper[0][1];
        complex128 u10 = oper[1][0];
        complex128 u11 = oper[1][1];
        std::vector<complex128> new_state = apply_single_qubit_noise_operator(state, u00, u01, u10, u11, dim, stride);
        double prob = compute_probability(new_state.data(), dim);
        probability_list[counter] = prob;
        counter++;
    }
    std::discrete_distribution<> d(probability_list.begin(), probability_list.end());
    int c = d(global_engine);
    auto it = std::next(noise_operators.begin(), c);
    const auto& oper = *it;
    complex128 u00 = oper[0][0];
    complex128 u01 = oper[0][1];
    complex128 u10 = oper[1][0];
    complex128 u11 = oper[1][1];
    apply_inplace_operator_1q(state, u00, u01, u10, u11, dim, stride);
    const double p_norm = 1 / std::sqrt(probability_list[c]);
    for (std::size_t i = 0; i < dim; i++){
        state[i] *= p_norm;
    }
}

static inline std::vector<complex128> apply_two_qubit_noise_operator(const complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u02, const complex128& __restrict__ u03, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const complex128& __restrict__ u12, const complex128& __restrict__ u13, const complex128& __restrict__ u20, const complex128& __restrict__ u21, const complex128& __restrict__ u22, const complex128& __restrict__ u23, const complex128& __restrict__ u30, const complex128& __restrict__ u31, const complex128& __restrict__ u32, const complex128& __restrict__ u33, const std::size_t dim, const std::size_t iters, const std::size_t m1, const std::size_t m2, const std::size_t ull_q1, const std::size_t ull_q2, const std::size_t target_mask){
    std::vector<complex128> new_state = std::vector<complex128>(dim, complex128(0.0, 0.0));
    #pragma omp parallel for
    for (std::size_t i = 0; i < iters; ++i){
        const std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        const std::size_t idx00 = pos;
        const std::size_t idx01 = pos | ull_q2;
        const std::size_t idx10 = pos | ull_q1;
        const std::size_t idx11 = pos | target_mask;

        const complex128 s00 = state[idx00];
        const complex128 s01 = state[idx01];
        const complex128 s10 = state[idx10];
        const complex128 s11 = state[idx11];

        new_state[idx00] += u00 * s00 + u01 * s01 + u02 * s10 + u03 * s11;
        new_state[idx01] += u10 * s00 + u11 * s01 + u12 * s10 + u13 * s11;
        new_state[idx10] += u20 * s00 + u21 * s01 + u22 * s10 + u23 * s11;
        new_state[idx11] += u30 * s00 + u31 * s01 + u32 * s10 + u33 * s11;
    }
    return new_state;  
}

static inline void apply_inplace_operator_2q(complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u02, const complex128& __restrict__ u03, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const complex128& __restrict__ u12, const complex128& __restrict__ u13, const complex128& __restrict__ u20, const complex128& __restrict__ u21, const complex128& __restrict__ u22, const complex128& __restrict__ u23, const complex128& __restrict__ u30, const complex128& __restrict__ u31, const complex128& __restrict__ u32, const complex128& __restrict__ u33, const std::size_t dim, const std::size_t iters, const std::size_t m1, const std::size_t m2, const std::size_t ull_q1, const std::size_t ull_q2, const std::size_t target_mask){
    #pragma omp parallel for
    for (std::size_t i = 0; i < iters; ++i){
        const std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        const std::size_t idx00 = pos;
        const std::size_t idx01 = pos | ull_q2;
        const std::size_t idx10 = pos | ull_q1;
        const std::size_t idx11 = pos | target_mask;

        const complex128 s00 = state[idx00];
        const complex128 s01 = state[idx01];
        const complex128 s10 = state[idx10];
        const complex128 s11 = state[idx11];

        state[idx00] = u00 * s00 + u01 * s01 + u02 * s10 + u03 * s11;
        state[idx01] = u10 * s00 + u11 * s01 + u12 * s10 + u13 * s11;
        state[idx10] = u20 * s00 + u21 * s01 + u22 * s10 + u23 * s11;
        state[idx11] = u30 * s00 + u31 * s01 + u32 * s10 + u33 * s11;
    }
}

static inline void apply_two_qubit_noise(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const std::vector<matrix>& noise_operators){
    const int num_operators = noise_operators.size();
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;
    std::vector<double> probability_list(num_operators, 0.0);
    int counter = 0;
    for (const matrix& oper : noise_operators){
        complex128 u00 = oper[0][0];
        complex128 u01 = oper[0][1];
        complex128 u02 = oper[0][2];
        complex128 u03 = oper[0][3];
        complex128 u10 = oper[1][0];
        complex128 u11 = oper[1][1];
        complex128 u12 = oper[1][2];
        complex128 u13 = oper[1][3];
        complex128 u20 = oper[2][0];
        complex128 u21 = oper[2][1];
        complex128 u22 = oper[2][2];
        complex128 u23 = oper[2][3];
        complex128 u30 = oper[3][0];
        complex128 u31 = oper[3][1];
        complex128 u32 = oper[3][2];
        complex128 u33 = oper[3][3];
        std::vector<complex128> new_state = apply_two_qubit_noise_operator(state, u00, u01, u02, u03, u10, u11, u12, u13, u20, u21, u22, u23, u30, u31, u32, u33, dim, iters, m1, m2, ull_q1, ull_q2, target_mask);
        double prob = compute_probability(new_state.data(), dim);
        probability_list[counter] = prob;
        counter++;
    }
    std::discrete_distribution<> d(probability_list.begin(), probability_list.end());
    int c = d(global_engine);
    auto it = std::next(noise_operators.begin(), c);
    const auto& oper = *it;
    complex128 u00 = oper[0][0];
    complex128 u01 = oper[0][1];
    complex128 u02 = oper[0][2];
    complex128 u03 = oper[0][3];
    complex128 u10 = oper[1][0];
    complex128 u11 = oper[1][1];
    complex128 u12 = oper[1][2];
    complex128 u13 = oper[1][3];
    complex128 u20 = oper[2][0];
    complex128 u21 = oper[2][1];
    complex128 u22 = oper[2][2];
    complex128 u23 = oper[2][3];
    complex128 u30 = oper[3][0];
    complex128 u31 = oper[3][1];
    complex128 u32 = oper[3][2];
    complex128 u33 = oper[3][3];
    apply_inplace_operator_2q(state, u00, u01, u02, u03, u10, u11, u12, u13, u20, u21, u22, u23, u30, u31, u32, u33, dim, iters, m1, m2, ull_q1, ull_q2, target_mask);
    const double p_norm = 1 / std::sqrt(probability_list[c]);
    #pragma omp parallel for
    for (std::size_t i = 0; i < dim; i++){
        state[i] *= p_norm;
    }
}

static inline void apply_X_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    #pragma omp parallel for
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = s1;
        state[j] = s0;
    }
}

static inline void apply_RZ_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);
    
    #pragma omp parallel for
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = complex128(cosine * s0.real() + sine * s0.imag(), cosine * s0.imag() - sine * s0.real());
        state[j] = complex128(cosine * s1.real() - sine * s1.imag(), sine * s1.real() + cosine * s1.imag());
    }
}

static inline void apply_RX_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    #pragma omp parallel for
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = complex128(cosine * s0.real() + sine * s1.imag(), cosine * s0.imag() - sine * s1.real());
        state[j] = complex128(sine * s0.imag() + cosine * s1.real(), cosine * s1.imag() - sine * s0.real());
    }
}

static inline void apply_SX_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    constexpr complex128 post_i = complex128(0.5, 0.5);
    constexpr complex128 negt_i = complex128(0.5, -0.5);

    #pragma omp parallel for
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;

        const complex128 s0 = state[i];
        const complex128 s1 = state[j];

        state[i] = post_i * s0 + negt_i * s1;
        state[j] = negt_i * s0 + post_i * s1;
    }
}

static inline void apply_CZ_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t m1 = (1ULL << (q1 < q2 ? q1 : q2)) - 1;
    const std::size_t m2 = (1ULL << ((q1 > q2 ? q1 : q2) - 1)) - 1;
    const std::size_t target_mask = (1ULL << q1) | (1ULL << q2);

    #pragma omp parallel for
    for (std::size_t i = 0; i < iters; ++i){
        std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        state[pos | target_mask] = -state[pos | target_mask];
    }
}

static inline void apply_RZZ_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;

    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    #pragma omp parallel for
    for (std::size_t i = 0; i < iters; ++i){
        std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        const std::size_t idx00 = pos;
        const std::size_t idx01 = pos | ull_q2;
        const std::size_t idx10 = pos | ull_q1;
        const std::size_t idx11 = pos | target_mask;

        const complex128 s00 = state[idx00];
        const complex128 s01 = state[idx01];
        const complex128 s10 = state[idx10];
        const complex128 s11 = state[idx11];

        state[idx00] = complex128(cosine * s00.real() + sine * s00.imag(), cosine * s00.imag() - sine * s00.real());
        state[idx01] = complex128(cosine * s01.real() - sine * s01.imag(), cosine * s01.imag() + sine * s01.real());
        state[idx10] = complex128(cosine * s10.real() - sine * s10.imag(), cosine * s10.imag() + sine * s10.real());
        state[idx11] = complex128(cosine * s11.real() + sine * s11.imag(), cosine * s11.imag() - sine * s11.real());
    }
}

static inline void apply_ECR_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;
    constexpr double inv_sqrt_2 = 0.7071067811865476;
    constexpr complex128 inv_sqrt_2i = complex128(0.0, 7071067811865476);

    #pragma omp parallel for
    for (std::size_t i = 0; i < iters; ++i){
        const std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const std::size_t pos = (i_s1 & m2) | ((i & ~m2) << 1);
        const std::size_t idx00 = pos;
        const std::size_t idx01 = pos | ull_q2;
        const std::size_t idx10 = pos | ull_q1;
        const std::size_t idx11 = pos | target_mask;

        const complex128 s00 = state[idx00];
        const complex128 s01 = state[idx01];
        const complex128 s10 = state[idx10];
        const complex128 s11 = state[idx11];

        state[idx00] = inv_sqrt_2 * s01 + inv_sqrt_2i * s11;
        state[idx01] = inv_sqrt_2 * s00 - inv_sqrt_2i * s10;
        state[idx10] = inv_sqrt_2 * s11 + inv_sqrt_2i * s01;
        state[idx11] = inv_sqrt_2 * s10 - inv_sqrt_2i * s00;
    }
}

void pure_state(const std::list<ItemEntry> instruction_list, const std::size_t num_qubits, complex128* __restrict__ state){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double)> gate_map;
    gate_map["x"] = apply_X_gate;
    gate_map["rz"] = apply_RZ_gate;
    gate_map["rx"] = apply_RX_gate;
    gate_map["sx"] = apply_SX_gate;
    gate_map["cz"] = apply_CZ_gate;
    gate_map["rzz"] = apply_RZZ_gate;
    gate_map["ecr"] = apply_ECR_gate;
    const std::size_t dim = std::size_t{1} << num_qubits;

    #pragma omp target data map(to: instruction_list, gate_map) map(tofrom: state[0:dim])
    {
        for (const ItemEntry& instruction : instruction_list){
            const std::string& gate_name = instruction.gate_name;
            const std::vector<std::size_t>& qubits = instruction.qubits;
            const double params = instruction.params;
            gate_map[gate_name](state, qubits[0], qubits[1], num_qubits, params);
        }
    }
}

void simulate_circuit(){}

PYBIND11_MODULE(simulator_gpu, m){
    m.doc() = "Module for simulating quantum circuits with noise on GPUs";
    m.def("simulate_circuit", &simulate_circuit, "Simulate a quantum circuit with noise on GPU");
}