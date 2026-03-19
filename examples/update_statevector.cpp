#include <complex>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
// #include <fstream>
// #include <iomanip>

using complex128 = std::complex<double>;
using complexVector = std::vector<complex128>;

void apply_H_gate(complex128* __restrict__ state,
                  const size_t q,
                  const size_t num_qubits) {
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    const size_t step = stride << 1;
    constexpr double inv_sqrt_2 = 0.7071067811865475244;

    for (size_t g = 0; g < dim; g += step) {
        for (size_t i = g; i < g + stride; i++) {
            const complex128 s0 = state[i];
            const complex128 s1 = state[i + stride];
            state[i] = inv_sqrt_2 * (s0 + s1);
            state[i + stride] = inv_sqrt_2 * (s0 - s1);
        }
    }
}

void apply_RX_gate(complex128* __restrict__ state,
                   const size_t q,
                   const size_t num_qubits,
                   double theta) {
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    const size_t step = stride << 1;

    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    for (size_t g = 0; g < dim; g += step) {
        for (size_t i = g; i < g + stride; i++) {
            const complex128 s0 = state[i];
            const complex128 s1 = state[i + stride];

            state[i] = complex128(cosine * s0.real() + sine * s1.imag(), cosine * s0.imag() - sine * s1.real());
            state[i + stride] = complex128(sine * s0.imag() + cosine * s1.real(), -sine * s0.real() + cosine * s1.imag());
        }
    }
}

void apply_CZ_gate_inplace(complex128* const __restrict__ state,
                           const size_t q1,
                           const size_t q2,
                           const size_t num_qubits) {

    const size_t dim = size_t{1} << num_qubits;
    const size_t stride_1 = size_t{1} << q1;
    const size_t stride_2 = size_t{1} << q2;
    const size_t step_1 = stride_1 << 1;
    const size_t step_2 = stride_2 << 1;

    for (size_t g2 = 0; g2 < dim; g2 += step_2) {
        for (size_t g1 = g2; g1 < g2 + stride_2; g1 += step_1) {
            const size_t start = g1 + stride_1 + stride_2;
            for (size_t i = start; i < start + stride_1; i++) {
                state[i] = -state[i];
            }
        }
    }
}

int main(int argc, char** argv) {
    size_t num_qubits = 25;
    unsigned int depth = 100;
    bool print_output = false;

    if (argc > 1) {
        num_qubits = static_cast<size_t>(std::stoull(argv[1]));
    }
    if (argc > 2) {
        depth = static_cast<unsigned int>(std::stoul(argv[2]));
    }

    const size_t dim = size_t{1} << num_qubits;
    complexVector state(dim, complex128{0.0, 0.0});
    state[0] = complex128{1.0, 0.0};

    std::mt19937 gen(1234567u);
    std::uniform_real_distribution<double> dis(-2.0 * M_PI, 2.0 * M_PI);

    std::vector<double> angles(num_qubits * depth);
    for (double& angle : angles) {
        angle = dis(gen);
    };

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma GCC ivdep
    for (size_t q = 0; q < num_qubits; ++q) {
        apply_H_gate(state.data(), q, num_qubits);
    }

    #pragma GCC ivdep
    for (unsigned int d = 0; d < depth; ++d) {
        for (size_t q = 0; q < num_qubits; ++q) {
            apply_RX_gate(state.data(), q, num_qubits, angles[num_qubits * d + q]);
        }
        for (size_t q = 0; q + 1 < num_qubits; ++q) {
            apply_CZ_gate_inplace(state.data(), q, q + 1, num_qubits);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    double seconds = duration * 1e-9;
    std::cout << "Time taken: " << seconds << " seconds." << std::endl;

    return 0;
}
