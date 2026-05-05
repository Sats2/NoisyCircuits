#include <vector>
#include <complex>
#include <omp.h>
#include <random>

using complex128 = std::complex<double>;
using uint = const unsigned short;
using matrix = std::vector<std::vector<complex128>>;

static inline double compute_probability(const complex128* __restrict__ state, const std::size_t dim, uint thread_count){
    double probability = 0.0;
    #pragma omp parallel for reduction(+:probability) num_threads(thread_count)
    for (std::size_t i = 0; i < dim; i++){
        probability += state[i].real() * state[i].real()  + state[i].imag() * state[i].imag();
    }
    return probability;
}

static inline std::vector<complex128> apply_single_qubit_noise_operator(const complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const std::size_t dim, const std::size_t stride, uint thread_count){
    std::vector<complex128> new_state = std::vector<complex128>(dim, complex128(0.0, 0.0));
    #pragma omp parallel for num_threads(thread_count)
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

static inline void apply_inplace_operator_1q(complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const std::size_t dim, const std::size_t stride, uint thread_count){
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = u00 * s0 + u01 * s1;
        state[j] = u10 * s0 + u11 * s1;
    }
}

static inline void apply_single_qubit_noise(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const std::vector<matrix>& noise_operators, const std::size_t dim, const std::size_t stride, std::mt19937_64& traj_engine, uint thread_count){
    int counter = 0;
    std::vector<double> probability_list(noise_operators.size(), 0.0);
    for (const matrix& oper : noise_operators){
        complex128 u00 = oper[0][0];
        complex128 u01 = oper[0][1];
        complex128 u10 = oper[1][0];
        complex128 u11 = oper[1][1];
        std::vector<complex128> new_state = apply_single_qubit_noise_operator(state, u00, u01, u10, u11, dim, stride, thread_count);
        double prob = compute_probability(new_state.data(), dim, thread_count);
        probability_list[counter] = prob;
        counter++;
    }
    std::discrete_distribution<> d(probability_list.begin(), probability_list.end());
    int c = d(traj_engine);
    auto it = std::next(noise_operators.begin(), c);
    const auto& oper = *it;
    complex128 u00 = oper[0][0];
    complex128 u01 = oper[0][1];
    complex128 u10 = oper[1][0];
    complex128 u11 = oper[1][1];
    apply_inplace_operator_1q(state, u00, u01, u10, u11, dim, stride, thread_count);
    const double p_norm = 1 / std::sqrt(probability_list[c]);
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t i = 0; i < dim; i++){
        state[i] *= p_norm;
    }
}