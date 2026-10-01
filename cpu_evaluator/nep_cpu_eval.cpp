// Binary adapter for NEP_CPU with checked, reduced neighbor storage capacity.
#include "nep.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>

template<class T> void read(std::ifstream& f, std::vector<T>& a) {
  f.read(reinterpret_cast<char*>(a.data()), a.size()*sizeof(T));
  if (!f) throw std::runtime_error("Incomplete binary input");
}
template<class T> void write(std::ofstream& f, const std::vector<T>& a) {
  f.write(reinterpret_cast<const char*>(a.data()), a.size()*sizeof(T));
  if (!f) throw std::runtime_error("Cannot write output");
}
int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("Usage: nep_cpu_eval model input.bin output.bin");
    omp_set_num_threads(2);
    std::ifstream in(argv[2], std::ios::binary);
    std::int32_t n=0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in || n <= 0 || n > 1000000) throw std::runtime_error("Invalid atom count");
    std::vector<double> box(9), pos(3ull*n), energy(n), force(3ull*n), virial(9ull*n);
    std::vector<int> type(n);
    read(in, box); read(in, type); read(in, pos);
    NEP nep(argv[1]);
    nep.compute(type, box, pos, energy, force, virial);
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    write(out, energy); write(out, force); write(out, virial);
    std::cout << "EVALUATED " << n << " atoms on CPU\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
