#ifdef PAIR_CLASS
// clang-format off
PairStyle(latte,PairLatte);
// clang-format on
#else

#ifndef LMP_PAIR_LATTE_H
#define LMP_PAIR_LATTE_H

#include "pair.h"
#include <string>
#include <fstream>

namespace LAMMPS_NS {

class PairLatte : public Pair {
 public:
  PairLatte(class LAMMPS *);
  ~PairLatte() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  struct block_params{
    int terms;
    int order;
    int dims;
    int indsize;
  };

 protected:
  int get_parameters(char*);
  int get_input_line(std::ifstream*, std::string*, std::string*);
  void RBF(float, float, float, float*, float*);

  int Nspecies, Nlayers;
  int activation;
  int format;
  bool cont_sp;
  std::string wfile_string;
  int* arch;
  int NdescrBlocks;
  std::vector<block_params> Blocks;
  std::vector<std::vector<int>> NumElemDims, CumElemDims, ElemFreq;
  double rcut;
  // Weights are stored as float32, not worth using double...
  float sigma;
  float **weights, **biases;
  float **descrCenters, **descrSigmas, **descrSpWei;


  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
