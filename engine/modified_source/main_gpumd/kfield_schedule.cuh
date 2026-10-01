// Scheduled increments for the previously validated per-atom move_field path.
#pragma once
#include "integrate/integrate.cuh"
#include "utilities/error.cuh"
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

static __global__ void set_scheduled_field_velocity(
  int count, int n, const int* index, const double* basis,
  double dki_over_dt, double dkii_over_dt, double* velocity)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < count) {
    int atom = index[q];
    for (int d = 0; d < 3; ++d) {
      velocity[atom + d*n] =
        basis[6*q+d]*dki_over_dt + basis[6*q+3+d]*dkii_over_dt;
    }
  }
}

class KFieldSchedule
{
public:
  bool enabled = false;
  bool used = false;
  std::string basis_path, schedule_path;
  std::vector<double> dki, dkii;
  GPU_Vector<int> indices;
  GPU_Vector<double> basis;
  int count = 0, n = 0;

  void parse(const char** param, int num_param)
  {
    if (num_param != 3 || enabled) {
      PRINT_INPUT_ERROR("Use kfield_schedule basis_file schedule_file once.");
    }
    enabled = true;
    basis_path = param[1];
    schedule_path = param[2];
  }

  void initialize(Integrate& integrate, Atom& atom, std::vector<Group>& groups, int steps)
  {
    if (!enabled) return;
    if (used || integrate.field_move_group < 0 ||
        (integrate.type != 0 && integrate.type != 22)) {
      PRINT_INPUT_ERROR("kfield_schedule needs one run, move_field, and nve or heat_lan.");
    }
    used = true;
    n = atom.number_of_atoms;
    const Group& group = groups[integrate.field_move_grouping_method];
    int expected = group.cpu_size[integrate.field_move_group];
    int file_n;
    std::ifstream input(basis_path);
    if (!(input >> file_n >> count) || file_n != n || count != expected || count <= 0) {
      PRINT_INPUT_ERROR("Invalid scheduled K-field basis header or boundary atom count.");
    }
    std::vector<int> ids(count), seen(n, 0);
    std::vector<double> values(6*count);
    for (int q = 0; q < count; ++q) {
      if (!(input >> ids[q]) || ids[q] < 0 || ids[q] >= n || seen[ids[q]] ||
          group.cpu_label[ids[q]] != integrate.field_move_group) {
        PRINT_INPUT_ERROR("Invalid or repeated scheduled boundary atom index.");
      }
      seen[ids[q]] = 1;
      for (int d = 0; d < 6; ++d) {
        if (!(input >> values[6*q+d]) || !std::isfinite(values[6*q+d])) {
          PRINT_INPUT_ERROR("Nonfinite or incomplete K-field basis.");
        }
      }
    }
    std::string extra;
    if (input >> extra) PRINT_INPUT_ERROR("Extra data in K-field basis.");
    input.close();
    input.open(schedule_path);
    int file_steps;
    if (!(input >> file_steps) || file_steps != steps) {
      PRINT_INPUT_ERROR("K-field schedule length must equal run steps.");
    }
    dki.resize(steps);
    dkii.resize(steps);
    for (int s = 0; s < steps; ++s) {
      if (!(input >> dki[s] >> dkii[s]) || !std::isfinite(dki[s]) || !std::isfinite(dkii[s])) {
        PRINT_INPUT_ERROR("Nonfinite or incomplete K-field schedule.");
      }
    }
    if (input >> extra) PRINT_INPUT_ERROR("Extra data in K-field schedule.");
    indices.resize(count);
    indices.copy_from_host(ids.data());
    basis.resize(6*count);
    basis.copy_from_host(values.data());
    printf("Scheduled K-field: %d boundary atoms, %d increments.\n", count, steps);
  }

  void apply(int step, double time_step, Integrate& integrate)
  {
    if (!enabled) return;
    // The secant velocity integrates each prescribed displacement increment exactly.
    set_scheduled_field_velocity<<<(count+127)/128,128>>>(
      count, n, indices.data(), basis.data(), dki[step]/time_step, dkii[step]/time_step,
      integrate.ensemble->field_velocity_per_atom.data());
    GPU_CHECK_KERNEL
  }
};

static KFieldSchedule kfield_schedule;
