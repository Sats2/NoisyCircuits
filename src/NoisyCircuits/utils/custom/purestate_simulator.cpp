#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

namespace py = pybind11;
using complex128 = std::complex<double>;


void apply_X_gate(complex128* __restrict__ state, const size_t q, const size_t num_qubits){
    /*
    This function applies the X gate to the q-th qubit in an n-qubit quantum state.

    Input:
        - state: Pointer to quantum state vector (size 2^n)
        - q: Index of the qubit to which the X gate is applied.
        - num_qubits: Total number of qubits in the system.
    */
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    const size_t step = stride << 1;

    for (size_t g = 0; g < dim; g += step){
        for (size_t i = g; i < g + stride; i++){
            const complex128 s0 = state[i];
            const complex128 s1 = state[i + stride];
            state[i] = s1;
            state[i + stride] = s0;
        }
    }
}

void apply_RZ_gate(complex128* __restrict__ state, const size_t q, const size_t num_qubits, double theta){
    /*
    This function applies the RZ gate to the q-th qubit in an n-qubit quantum state.

    Input:
        - state: Pointer to quantum state vector (size 2^n)
        - q: Index of the qubit to which the RZ gate is applied.
        - num_qubits: Total number of qubits in the system.
        - theta: Rotation angle for the RZ gate.
    */
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    const size_t step = stride << 1;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    for (size_t g = 0; g < dim; g += step){
        for (size_t i = g; i < g + stride; i++){
            const complex128 s0 = state[i];
            const complex128 s1 = state[i + stride];
            state[i] = complex128(cosine * s0.real() + sine * s0.real(), cosine * s0.imag() + sine * s0.real());
            state[i + stride] = complex128(cosine * s1.real() - sine * s1.imag(), sine * s1.real() + cosine * s1.imag());
        }
    }
}

void apply_RX_gate(complex128* __restrict__ state,
                   const size_t q,
                   const size_t num_qubits,
                   double theta){
    /*
    This function applies the RX gate to the q-th qubit in an n-qubit quantum state.

    Input:
        - state: Pointer to quantum state vector (size 2^n)
        - q: Index of the qubit to which the RX gate is applied.
        - num_qubits: Total number of qubits in the system.
        - theta: Rotation angle for the RX gate.
    */
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    const size_t step = stride << 1;

    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    for (size_t g = 0; g < dim; g+= step){
        for (size_t i = g; i < g + stride; i++){
            const complex128 s0 = state[i];
            const complex128 s1 = state[i + stride];

            state[i] = complex128(cosine * s0.real() + sine * s1.imag(), cosine * s0.imag() - sine * s1.real());
            state[i + stride] = complex128(sine * s0.imag() + cosine * s1.real(), -sine * s0.real() + cosine * s1.imag());
        }
    }
}

void apply_SX_gate(complex128* __restrict__ state, const size_t q, const size_t num_qubits){
    /*
    This function applies the SX gate to the q-th qubit in an n-qubit quantum state.

    Input:
        - state: Pointer to quantum state vector (size 2^n)
        - q: Index of the qubit to which the SX gate is applied.
        - num_qubits: Total number of qubits in the system.
    */
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    const size_t step = stride << 1;
    constexpr complex128 post_i = complex128(0.5, 0.5);
    constexpr complex128 negt_i = complex128(0.5, -0.5);

    for (size_t g = 0; g < dim; g += step){
        for (size_t i = g; i < g + stride; i++){
            const complex128 s0 = state[i];
            const complex128 s1 = state[i + stride];
            state[i] = post_i * s0 + negt_i * s1;
            state[i + stride] = negt_i * s0 + post_i * s1;
        }
    }
}

void apply_CZ_gate(complex128* const __restrict__ state, const size_t q1, const size_t q2, const size_t num_qubits){
    /*
    This function applies the CZ gate to the q1-th and q2-th qubit in an n-qubit quantum state. Follows the Little Endian convention.

    Input:
        - state: Pointer to quantum state vector (size 2^n)
        - q1: Index of the control qubit for the CZ gate.
        - q2: Index of the target qubit for the CZ gate.
        - num_qubits: Total number of qubits in the system.
    */
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride_1 = size_t{1} << q1;
    const size_t stride_2 = size_t{1} << q2;
    const size_t step_1 = stride_1 << 1;
    const size_t step_2 = stride_2 << 1;

    for (size_t g2 = 0; g2 < dim; g2 += step_2){
        for (size_t g1 = g2; g1 < g2 + stride_2; g1 += step_1){
            const size_t start = g1 + stride_1 + stride_2;
            for (size_t i = start; i < start + stride_1; i++){
                state[i] = -state[i];
            }
        }
    }
}

void apply_RZZ_gate(complex128* const __restrict__ state, const size_t q1, const size_t q2, const size_t num_qubits, double theta){
    /*
    This function applies the RZZ gate to the q1-th and q2-th qubit in an n-qubit quantum state. Follows the Little Endian convention.

    Input:
        - state: Pointer to quantum state vector (size 2^n)
        - q1: Index of the control qubit for the RZZ gate.
        - q2: Index of the target qubit for the RZZ gate.
        - num_qubits: Total number of qubits in the system.
        - theta: Rotation angle for the RZZ gate.
    */
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride_1 = size_t{1} << q1;
    const size_t stride_2 = size_t{1} << q2;
    const size_t step_1 = stride_1 << 1;
    const size_t step_2 = stride_2 << 1;

    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    for (size_t g2 = 0; g2 < dim; g2 += step_2){
        for (size_t g1 = g2; g1 < g2 + stride_2; g1 += step_1){
            for (size_t i = g1; i < g1 + stride_1; i++){
                const complex128 s00 = state[i];
                const complex128 s01 = state[i + stride_2];
                const complex128 s10 = state[i + stride_1];
                const complex128 s11 = state[i + stride_1 + stride_2];
                state[i] = complex128(cosine * s00.real() + sine * s00.imag(), cosine * s00.imag() - sine * s00.real());
                state[i + stride_1] = complex128(cosine * s01.real() - sine * s01.imag(), sine * s01.real() + cosine * s01.imag());
                state[i + stride_2] = complex128(cosine * s10.real() - sine * s10.imag(), sine * s10.real() + cosine * s10.imag());
                state[i + stride_1 + stride_2] = complex128(cosine * s11.real() + sine * s11.imag(), cosine * s11.imag() - sine * s11.real());
            }
        }
    }
}

void apply_ECR_gate(complex128* const __restrict__ state, const size_t q1, const size_t q2, const size_t num_qubits){
    /*
    This function applies the ECR gate to the q1-th and q2-th qubit in an n-qubit quantum state. Follows the Little Endian convention.

    Input:
        - state: Pointer to quantum state vector (size 2^n)
        - q1: Index of the control qubit for the ECR gate.
        - q2: Index of the target qubit for the ECR gate.
        - num_qubits: Total number of qubits in the system.
    */
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride_1 = size_t{1} << q1;
    const size_t stride_2 = size_t{1} << q2;
    const size_t step_1 = stride_1 << 1;
    const size_t step_2 = stride_2 << 1;

    constexpr double inv_sqrt_2 = 0.7071067811865476;
    constexpr complex128 inv_sqrt_2i = complex128(0.0, 0.7071067811865476);

    for (size_t g2 = 0; g2 < dim; g2 += step_2){
        for (size_t g1 = g2; g1 < g2 + stride_2; g1 += step_1){
            for (size_t i = g1; i < g1 + stride_1; i++){
                const complex128 s00 = state[i];
                const complex128 s01 = state[i + stride_1];
                const complex128 s10 = state[i + stride_2];
                const complex128 s11 = state[i + stride_1 + stride_2];
                state[i] = inv_sqrt_2 * s01 + inv_sqrt_2i * s11;
                state[i + stride_1] = inv_sqrt_2 * s00 - inv_sqrt_2i * s10;
                state[i + stride_2] = inv_sqrt_2 * s11 + inv_sqrt_2i * s01;
                state[i + stride_1 + stride_2] = inv_sqrt_2 * s10 - inv_sqrt_2i * s00;
            }
        }
    }
}