#include <complex>
#include <vector>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using complex128 = std::complex<double>;
using complexVector = std::vector<complex128>;
using uint8 = const unsigned short;


void set_thread_count(uint8 thread_count){
    omp_set_num_threads(thread_count);
}

void apply_H_gate(complex128* __restrict__ state,
                  const size_t q,
                  const size_t num_qubits){
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    // const size_t step = stride << 1;
    constexpr double inv_sqrt_2 = 0.70710678118655;

    #pragma omp parallel for
    for (size_t pair = 0; pair < (dim >> 1); ++pair){
        const size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = inv_sqrt_2 * (s0 + s1);
        state[i + stride] = inv_sqrt_2 * (s0 - s1);
    }
}

void apply_RX_gate(complex128* __restrict__ state,
                   const size_t q,
                   const size_t num_qubits,
                   double theta){
    const size_t dim = size_t{1} << num_qubits;
    const size_t stride = size_t{1} << q;
    // const size_t step = stride << 1;

    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    #pragma omp parallel for
    for (size_t pair = 0; pair < (dim >> 1); ++pair){
        const size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];

        state[i] = complex128(cosine * s0.real() + sine * s1.imag(), cosine * s0.imag() - sine * s1.real());
        state[i + stride] = complex128(sine * s0.imag() + cosine * s1.real(), -sine * s0.real() + cosine * s1.imag());
    }
}

void apply_CZ_gate(complex128* const __restrict__ state, 
                    const size_t q1,
                    const size_t q2,
                    const size_t num_qubits){
    const size_t dim = size_t{1} << num_qubits;
    size_t iters = dim >> 2;
    const size_t q_min = q1 < q2 ? q1 : q2;
    const size_t q_max = q1 > q2 ? q1 : q2;
    const size_t m1 = (1ULL << q_min) - 1;
    const size_t m2 = (1ULL << (q_max - 1)) - 1;
    const size_t target_mask = (1ULL << q1) | (1ULL << q2);

    #pragma omp parallel for
    for (size_t i = 0; i < iters; ++i){
        size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        state[pos | target_mask] = -state[pos | target_mask];
    }
}


void run_circuit_kernel(complex128* __restrict__ state,
                        const double* __restrict__ angles_ptr,
                        const size_t num_qubits,
                        const unsigned int depth){
    for (size_t q = 0; q < num_qubits; q++){
        apply_H_gate(state, q, num_qubits);
    }

    for (unsigned int d = 0; d < depth; d++){
        for (size_t q = 0; q < num_qubits; q++){
            apply_RX_gate(state, q, num_qubits, angles_ptr[d * num_qubits + q]);
        }
        for (size_t q = 0; q < num_qubits - 1; q++){
            apply_CZ_gate(state, q, q + 1, num_qubits);
        }
    }
}


complexVector run_circuit(py::array_t<double>input_angles,
                            py::size_t num_qubits,
                        uint8 thread_count){
    set_thread_count(thread_count);
    py::buffer_info angles_buf = input_angles.request();
    double* angles_ptr = static_cast<double*>(angles_buf.ptr);
    size_t num_angles = angles_buf.size;
    unsigned int depth = num_angles / num_qubits;
    const size_t dim = size_t{1} << num_qubits;
    complexVector state(dim, complex128{0.0, 0.0});
    state[0] = complex128{1.0, 0.0};

    run_circuit_kernel(state.data(), angles_ptr, num_qubits, depth);

    return state;
}

py::array_t<complex128> run_circuit_new(py::array_t<double>input_angles, py::size_t num_qubits, uint8 thread_count){
    set_thread_count(thread_count);
    py::buffer_info angles_buf = input_angles.request();
    double* angles_ptr = static_cast<double*>(angles_buf.ptr);
    size_t num_angles = angles_buf.size;
    unsigned int depth = num_angles / num_qubits;
    const size_t dim = size_t{1} << num_qubits;
    py::array_t<complex128> out(dim);
    py::buffer_info output_buf = out.request();
    auto* state = static_cast<complex128*>(output_buf.ptr);
    std::fill(state, state + dim, complex128{0.0, 0.0});
    state[0] = complex128{1.0, 0.0};

    run_circuit_kernel(state, angles_ptr, num_qubits, depth);
    return out;
}

void run_circuit_inplace(py::array_t<double> input_angles,
                         py::size_t num_qubits,
                         py::array_t<complex128> out_state, uint8 thread_count){
    set_thread_count(thread_count);
    py::buffer_info angles_buf = input_angles.request();
    double* angles_ptr = static_cast<double*>(angles_buf.ptr);
    const size_t num_angles = angles_buf.size;
    const unsigned int depth = num_angles / num_qubits;
    const size_t dim = size_t{1} << num_qubits;

    py::buffer_info out_buf = out_state.request();
    if (static_cast<size_t>(out_buf.size) != dim){
        throw std::runtime_error("out_state has wrong size; expected 2**num_qubits");
    }

    auto* state = static_cast<complex128*>(out_buf.ptr);
    std::fill(state, state + dim, complex128{0.0, 0.0});
    state[0] = complex128{1.0, 0.0};

    run_circuit_kernel(state, angles_ptr, num_qubits, depth);
}


PYBIND11_MODULE(run, m){
    m.doc() = "C++ Module to simulate a Variational Circuit";
    m.def("run_circuit", &run_circuit, "Runs the Quantum Circuit Simulator");
    m.def("run_circuit_new", &run_circuit_new, "Runs the Quantum Circuit Simulator and returns a numpy array");
    m.def("run_circuit_inplace", &run_circuit_inplace, "Runs the Quantum Circuit Simulator and writes into a provided numpy output array");
}