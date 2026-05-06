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

struct pair_hash{
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct ItemEntry{
    std::string gate_name;
    std::vector<std::size_t> qubits;
    double params;
};

using complex128 = std::complex<double>;
using uint8 = const unsigned short;
using matrix = std::vector<std::vector<complex128>>;
using noise_map = std::unordered_map<std::string, std::vector<matrix>>;
using noise_map2q = std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash>>;