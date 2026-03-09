
#include "pair_latte.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

// #include "stdio.h"
// #include "stdlib.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLatte::PairLatte(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  // manybody_flag = 1; Maybe set this?
}

/* ---------------------------------------------------------------------- */

PairLatte::~PairLatte()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairLatte::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int inum = list->inum;
  assert(inum==nlocal); //Just a check..
  int *ilist = list->ilist;
  int *jlist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int i,j,k,a,b,p,e,l,o;
  double xtmp, ytmp, ztmp, dx, dy, dz, rtmp;

  //Looping to get the number of pairs
  int Npairs = 0;
  int CumPairs[inum];
  for (i=0; i<inum; i++) {
    CumPairs[i] = Npairs;
    Npairs += numneigh[ilist[i]];
  }

  //Looping over pairs to get a list
  int pair_i[Npairs]; // Local index
  int pair_abs_i[Npairs]; // Absolute index
  int pair_abs_j[Npairs];
  int type_i[Npairs];
  int type_j[Npairs];
  float vers_ij[Npairs*3];
  float rij[Npairs];
  for (i=0; i<inum; i++) {
    a = ilist[i];
    xtmp = x[a][0];
    ytmp = x[a][1];
    ztmp = x[a][2];
    jlist = firstneigh[a];
    for (j=0; j<numneigh[a]; j++){
      p = CumPairs[i] + j;
      b = jlist[j];
      b &= NEIGHMASK;
      pair_i[p] = i;
      pair_abs_i[p] = a;
      pair_abs_j[p] = b;
      type_i[p] = type[a]-1;
      type_j[p] = type[b]-1;
      dx = x[b][0] - xtmp;
      dy = x[b][1] - ytmp;
      dz = x[b][2] - ztmp;
      rtmp = sqrt(dx*dx + dy*dy + dz*dz);
      // Here we could double check the cutoff?
      // (but then could not rely on precomputed numneigh...)
      rij[p] = (float)rtmp;
      rtmp = 1.0/rtmp;
      vers_ij[3*p    ] = (float)(dx*rtmp);
      vers_ij[3*p + 1] = (float)(dy*rtmp);
      vers_ij[3*p + 2] = (float)(dz*rtmp);
    }
  }

  //Allocating final arrays
  int Dsize = arch[0];
  float descr[nlocal*Dsize];
  // std::fill_n(descr,nlocal*Dsize,0.0);
  float ddescr[Dsize*Npairs*3];

  int dshift = 0;
  //Looping over descriptor blocks
  for (b=0; b<NdescrBlocks; b++){
    //Allocating tmps for this descriptor
    block_params par = Blocks[b];
    int terms = par.terms;
    int order = par.order;
    int dims = par.dims;
    int size1 = terms*order;
    int indsize = par.indsize;
    int size2 = size1*indsize;
    std::vector<int> bElemFreq = ElemFreq[b];
    std::vector<int> bElemDims = NumElemDims[b];
    std::vector<int> bCumElemDims = CumElemDims[b];
    float Asummed[nlocal*size2];
    std::fill_n(Asummed,nlocal*size2,0.0);
    float dApair[Npairs*size2*3];

    //First loop over pairs: compute A contributions, A sums, dA contr.
    for (p=0; p<Npairs; p++){
      int shift2 = p*size2;
      //If there are indices, precompute the tensors:
      // Tensors for A terms
      float tens1[order*indsize];
      // Tensors for dA terms in R/r
      float tens2[order*indsize*3];
      std::fill_n(tens2,order*indsize*3,0.0);
      if (dims>0){
        // Loop order here?
        for (o=0; o<order;o++){
          int ord_shift = bCumElemDims[o];
          for (l=0; l<indsize; l++){
            tens1[o*indsize+l] = 1.0;
            // We loop on all indices
            for (e=0; e<bElemDims[o]; e++){
              int vers_shift = (l/bElemFreq[ord_shift+e])%3;
              tens1[o*indsize+l] *= vers_ij[3*p+vers_shift];
              float tmp = 1.0;
              // We loop on all other indices
              for (int ee=0; ee<bElemDims[o]; ee++){
                if (ee!=e) {
                  int vers_shift2 = (l/bElemFreq[ord_shift+ee])%3;
                  tmp *= vers_ij[3*p+vers_shift2];
                }
              }
              // Add the term with delta between e and final index
              tens2[3*(o*indsize+l)+vers_shift] += tmp;
            }
          }
        }
      }

      //Loop over descr terms
      for (e=0; e<size1; e++){
        float rf, drf;
        RBF(rij[p], descrCenters[b][type_i[p]*size1+e], \
            descrSigmas[b][type_i[p]*size1+e], &rf, &drf);
        rf *= descrSpWei[b][(type_i[p]*Nspecies+type_j[p])*size1+e];
        drf *= descrSpWei[b][(type_i[p]*Nspecies+type_j[p])*size1+e];
        if (dims==0){
          // This is only for '-':
          Asummed[pair_i[p]*size2+e] += rf;
          dApair[3*(shift2+e)  ] = drf*vers_ij[3*p  ];
          dApair[3*(shift2+e)+1] = drf*vers_ij[3*p+1];
          dApair[3*(shift2+e)+2] = drf*vers_ij[3*p+2];
        }
        else{
          int el_num = e/terms;
          float rfr = rf/rij[p];
          drf -= bElemDims[el_num]*rfr;
          for (l=0; l<indsize; l++){
            // We sum RBF and the tensor to the right atom
            Asummed[pair_i[p]*size2+e*indsize+l] += \
                                rf*tens1[el_num*indsize+l];
            dApair[3*(shift2+e*indsize+l)  ] = \
                                drf*tens1[el_num*indsize+l]*vers_ij[3*p  ] + \
                                rfr*tens2[3*(el_num*indsize+l)  ];
            dApair[3*(shift2+e*indsize+l)+1] = \
                                drf*tens1[el_num*indsize+l]*vers_ij[3*p+1] + \
                                rfr*tens2[3*(el_num*indsize+l)+1];
            dApair[3*(shift2+e*indsize+l)+2] = \
                                drf*tens1[el_num*indsize+l]*vers_ij[3*p+2] + \
                                rfr*tens2[3*(el_num*indsize+l)+2];
          }
        }
      }
    }

    // Loop over atoms to compute B
    for (i=0; i<nlocal; i++){
      //Loop over descr terms
      for (e=0; e<terms; e++){
        if (dims==0){
          // This is only for '-':
          descr[i*Dsize+dshift+e] = Asummed[i*size1+e];
        }
        else {
          descr[i*Dsize+dshift+e] = 0.0;
          for (l=0; l<indsize; l++){
            float tmp = 1.0;
            for (o=0; o<order;o++){
              tmp *= Asummed[i*size2+(e+o*terms)*indsize+l];
            }
            descr[i*Dsize+dshift+e] += tmp;
          }
        }
      }
    }

    //Second loop over pairs: producs to compute dB
    for (p=0; p<Npairs; p++){
      int shift1 = p*Dsize+dshift;
      int shift2 = p*size2;
      int shift3 = pair_i[p]*size2;
      //Loop over descr terms
      for (e=0; e<terms; e++){
        if (dims==0){
          // This is only for '-':
          ddescr[3*(shift1+e)  ] = dApair[3*(shift2+e)  ];
          ddescr[3*(shift1+e)+1] = dApair[3*(shift2+e)+1];
          ddescr[3*(shift1+e)+2] = dApair[3*(shift2+e)+2];
        }
        else {
          ddescr[3*(shift1+e)  ] = 0.0;
          ddescr[3*(shift1+e)+1] = 0.0;
          ddescr[3*(shift1+e)+2] = 0.0;
          // This loops over each term being the derivative
          // i.e. d(ABC) = dA BC + A dB C + AB dC etc..
          // Then I sum over all terms and all indices for contraction
          for (int od=0; od<order; od++){
            for (l=0; l<indsize; l++){
              float tmpx = 1.0;
              float tmpy = 1.0;
              float tmpz = 1.0;
              for (o=0; o<order; o++){
                if (o==od) {
                  tmpx *= dApair[3*(shift2+(e+o*terms)*indsize+l)];
                  tmpy *= dApair[3*(shift2+(e+o*terms)*indsize+l)+1];
                  tmpz *= dApair[3*(shift2+(e+o*terms)*indsize+l)+2];
                }
                else{
                  tmpx *= Asummed[shift3+(e+o*terms)*indsize+l];
                  tmpy *= Asummed[shift3+(e+o*terms)*indsize+l];
                  tmpz *= Asummed[shift3+(e+o*terms)*indsize+l];
                }
              }
              ddescr[3*(shift1+e)  ] += tmpx;
              ddescr[3*(shift1+e)+1] += tmpy;
              ddescr[3*(shift1+e)+2] += tmpz;
            }            
          }
        }
      }
    }
    // Keeping track of shift in final descriptor
    dshift += terms;
  }
  // Prints for test
  // for (k=0;k<Dsize;k++) std::cout << descr[k] << " ";
  // for (p=0; p<Npairs; p++){
  //   for (k=0;k<Dsize;k++) {
  //     for (j=0; j<3; j++) std::cout << ddescr[p*Dsize*3+k*3+j] << " ";
  //     std::cout << "\n";
  //   }  
  //   std::cout << "\n";
  // }
  // std::cout << "\n";

  // Computing the NN
  //Creating pointers we will use throughout
  float *input, *output;
  float *dinput, *doutput;
  // Nat, Dsize
  input = descr;

  //Looping over layers
  for (l=0; l<Nlayers; l++){
    int insize = arch[l];
    int outsize = arch[l+1];
    // [Nat, out_size]
    output = new float[nlocal*outsize];
    // [Nat, Dsize, out_size]
    doutput = new float[nlocal*outsize*Dsize];
    std::fill_n(doutput,nlocal*outsize*Dsize,0.0);
    // Loop over atoms
    for (a=0; a<nlocal; a++){
      // [out_size, in_size]
      float* atw = &weights[l][(type[ilist[a]]-1)*insize*outsize];
      float* atb = &biases[l][(type[ilist[a]]-1)*outsize];
      // Loop over output elements
      for (i=0; i<outsize; i++){
        float dtmp, dtmp2;
        output[a*outsize+i] = atb[i];
        // Loop over input elements for x
        for (j=0; j<insize; j++){
          output[a*outsize+i] += atw[i*insize+j]*input[a*insize+j];
        }
        // Nonlinearity
        if (l<Nlayers-1){
          // exp
          if(activation==0){
            dtmp = exp(-output[a*outsize+i]*output[a*outsize+i]);
            dtmp2 = -2.0*output[a*outsize+i]*dtmp;
            output[a*outsize+i] = dtmp;
          }
        }
        // Also loop over derivatives for dx
        for (k=0; k<Dsize; k++){
          // Special case for the first layer (dx_i/dx_j=delta_ij)
          if (l==0){
            doutput[(a*Dsize+k)*outsize+i] = atw[i*insize+k];
          }
          else {
            for (j=0; j<insize; j++){
              doutput[(a*Dsize+k)*outsize+i] += atw[i*insize+j]*dinput[(a*Dsize+k)*insize+j];
            }
          }
          // Nonlinearity
          if (l<Nlayers-1){
            // exp
            if(activation==0){
              doutput[(a*Dsize+k)*outsize+i] *= dtmp2;
            }
          }
        }
      }
    }
    // Let's delete if we allocated
    if (l!=0){
      delete[] input;
      delete[] dinput;
    }
    // Output is the new input
    input = output;
    dinput = doutput;
  }

  // Adding to total energy if required
  if (eflag_global || eflag_atom){
    for (a=0; a<nlocal; a++){
      if (eflag_global) eng_vdwl += (double)input[a];
      if (eflag_atom) eatom[ilist[a]] += (double)input[a];
    }
  }
  delete[] input;

  //Last loop over pairs: add force contributions to the correct atoms
  for (p=0; p<Npairs; p++){
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    int shift1 = p*Dsize;
    int shift2 = pair_i[p]*Dsize;
    // Loop over descriptor elements
    for (k=0; k<Dsize; k++){
      fx += dinput[shift2+k] * ddescr[3*(shift1+k)  ];
      fy += dinput[shift2+k] * ddescr[3*(shift1+k)+1];
      fz += dinput[shift2+k] * ddescr[3*(shift1+k)+2];
      // if (p==0) std::cout << "("<< dinput[shift2+k] * ddescr[3*(shift1+k)  ] << " "\
      //   << dinput[shift2+k] << " " << ddescr[3*(shift1+k)  ] << ") ";
    }
    // if (p==0) std::cout << "\n";
    // Assign to the correct atoms
    int ii = pair_abs_i[p];
    int jj = pair_abs_j[p];
    f[ii][0] += (double)fx;
    f[ii][1] += (double)fy;
    f[ii][2] += (double)fz;
    f[jj][0] -= (double)fx;
    f[jj][1] -= (double)fy;
    f[jj][2] -= (double)fz;
  }
  delete[] dinput;

  if (vflag_fdotr) virial_fdotr_compute();
}


void PairLatte::RBF(float x, float xc, float sigma, float *f, float *df)
{
  float y = (x-xc)/sigma;
  float t = 1.0 - y*y;
  if (t<0.0){
    *f = 0.0;
    *df = 0.0;
    return;
  }
  float q = t*t/(xc*xc);
  *f = t*q;
  *df = -6.0*y*q/sigma;
  return;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLatte::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 1;

  memory->create(cutsq, n, n, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLatte::settings(int narg, char **arg)
{
  if (narg != 0) {
    error->all(FLERR,"pair_latte requires no arguments.\n");
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

// Get a new line skipping comments or empty lines
// Fill key,value if 'key=value', return 1
// Return 0 if eof, <0 if error, >0 if okay
int PairLatte::get_input_line(std::ifstream* file, std::string* key, std::string* value){
  std::string line;
  int parsed = 0;
  while(!parsed){
    std::getline(*file,line);
    // Exit on EOF
    if(file->eof()) return 0;
    // Exit on bad read
    if(file->bad()) return -1;
    // Remove spaces
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    // Skip empty line
    if(line.length()==0) continue;
    // Skip comments
    if(line.at(0)=='#') continue;
    // Look for equal sign
    std::string eq = "=";
    size_t eqpos = line.find(eq);
    // Parse key-value pair
    if(eqpos != std::string::npos){
      *key = line.substr(0,eqpos);
      *value = line.substr(eqpos+1,line.length()-1);
      return 1;
    }
    std::cout << line << std::endl;
    parsed = 1;
  }
  return -1;
}

/* ----------------------------------------------------------------------
   read parameters file
------------------------------------------------------------------------- */

int PairLatte::get_parameters(char* directory){
  // Parsing the potential parameters
  std::ifstream params_file;
  std::ifstream weights_file;
  std::string key, value;
  std::string dir_string(directory);
  std::string file_string(dir_string+"/panna.in");
  std::string comma = ",";
  std::string colon = ":";
  std::string sqbo = "[";
  std::string sqbc = "]";
  std::string rbo = "(";
  std::string rbc = ")";

  // Initializing some parameters before reading:
  Nspecies = 0;
  Nlayers = 0;
  rcut = 0.0;
  sigma = 0.0;
  // 0 is binary format, 1 is torch (for kokkos)
  format = 0;
  // Whether to reorder atoms to have species contiguous
  cont_sp = false;

  // Opening file  
  params_file.open(file_string.c_str());
  if(params_file){
    int parseint = get_input_line(&params_file,&key,&value);
    while(parseint>0){
      // Parsing architecture
      if (key=="architecture"){
        size_t pos = value.find(comma);
        Nlayers = std::atoi(value.substr(0, pos).c_str())-1;
        arch = new int[Nlayers+1];
        value.erase(0, pos+1);
        for(int s=0; s<Nlayers+1; s++){
          pos = value.find(comma);
          arch[s] = std::atoi(value.substr(0, pos).c_str());
          value.erase(0, pos+1);
        }
      }
      // Parsing species (just Nsp for now)
      else if (key=="species"){
        size_t pos = value.find(comma);
        Nspecies = std::atoi(value.substr(0, pos).c_str());
      }
      // Parsing descriptor
      if (key=="descriptor_shape"){
        size_t pos = value.find(colon);
        // Number of blocks in the descriptor
        NdescrBlocks = std::atoi(value.substr(0, pos).c_str());
        value.erase(0, pos+1);
        for(int s=0; s<NdescrBlocks; s++){
          // Number of dimensions for each element
          std::vector<int> thisElemDims;
          // Beginning of dimension indices for each element
          std::vector<int> thisCumElemDims;
          // Period associated to each index
          std::vector<int> thisElemFreq;          
          thisCumElemDims.push_back(0);
          pos = value.find(colon);
          std::string block = value.substr(0, pos).c_str();
          // Reading N_terms
          size_t pos2 = block.find(comma);
          int terms = std::atoi(block.substr(0, pos2).c_str());
          block.erase(0, pos2+1);
          // Reading N_dims
          pos2 = block.find(comma);
          int dims = std::atoi(block.substr(0, pos2).c_str());
          // Computing the number of elements
          // TODO: use int power here
          int indsize = (int)pow(3,dims);
          block.erase(0, pos2+1);
          // Reading the number of bodies
          pos2 = block.find(sqbo);
          int order = std::atoi(block.substr(0, pos2).c_str());
          block.erase(0, pos2+1);
          block_params params = {terms, order, dims, indsize};
          Blocks.push_back(params);
          // Reading signatures only if there are signatures to read
          if (dims>0){
            // This was the way of storing things.. now we compute
            // ElemIndices[s] = new int**[NdescrBodies[s]];
            // Reading for each term
            for (int b=0; b<order; b++){
              pos2 = block.find(rbo);
              // Reading how many indices are in this element
              int numInds = std::atoi(block.substr(0, pos2).c_str());
              thisElemDims.push_back(numInds);
              thisCumElemDims.push_back(thisCumElemDims[b]+numInds);
              block.erase(0, pos2+1);
              // ElemIndices[s][b] = new int*[numInds];
              // Looping over indices
              for (int i=0; i<numInds; i++){
                pos2 = block.find(comma);
                int this_ind = std::atoi(block.substr(0, pos2).c_str());
                block.erase(0, pos2+1);
                // ElemIndices[s][b][i] = new int[NdescrDimElems[s]];
                // This indicates the frequency, where 0 is the first index etc..
                // TODO Use int pow
                int this_freq = (int)pow(3,dims-this_ind-1);
                thisElemFreq.push_back(this_freq);
                // for (int e=0; e<NdescrDimElems[s]; e++){
                  // We store the index [0,1,2]
                  // For each element of a NdescrDims[s] dimensional tensor
                  // at the frequency corresponding to index this_ind
                  // ElemIndices[s][b][i][e] = (e/this_freq)%3;
                // }
              }
              //Deleting '(,'
              pos2 = block.find(rbc);
              block.erase(0, pos2+2);
            }
          }
          NumElemDims.push_back(thisElemDims);
          CumElemDims.push_back(thisCumElemDims);
          ElemFreq.push_back(thisElemFreq);
          value.erase(0, pos+1);
        }
      }
      // Other values
      else if (key=="activation"){
        activation = std::atoi(value.c_str());}
      else if (key=="cutoff"){
        rcut = std::atof(value.c_str());}
      else if (key=="sigma"){
        sigma = std::atof(value.c_str());}
      else if (key=="model_file"){
        wfile_string = dir_string+"/"+value;}
      else if (key=="format"){
        format = std::atoi(value.c_str());}
      else if (key=="cont_sp"){
        cont_sp = std::atoi(value.c_str())>0;}
      // Get new line
      parseint = get_input_line(&params_file,&key,&value);
    }
  }
  else return -1;
  params_file.close();

  if (format==0){
    // Loading the descriptor and network if binary
    weights_file.open(wfile_string.c_str(), std::ios::binary);
    if(!weights_file.is_open()){
      std::cout << "Error reading weights file." << std::endl;
      return -3;
    }
    // Read descriptors block by block
    descrSpWei = new float*[NdescrBlocks];
    descrCenters = new float*[NdescrBlocks];
    descrSigmas = new float*[NdescrBlocks];
    for (int b=0; b<NdescrBlocks; b++){
      block_params par = Blocks[b];
      // Read species weights
      int Msize = Nspecies*Nspecies*par.terms*par.order;
      descrSpWei[b] = new float[Msize];
      weights_file.read(reinterpret_cast<char*>(descrSpWei[b]), Msize*sizeof(float));
      // Read centers
      Msize = Nspecies*par.terms*par.order;
      descrCenters[b] = new float[Msize];
      weights_file.read(reinterpret_cast<char*>(descrCenters[b]), Msize*sizeof(float));
      // Read sigmas if not constant, otherwise initialize as const
      Msize = Nspecies*par.terms*par.order;
      descrSigmas[b] = new float[Msize];
      if(sigma==0.0){
        weights_file.read(reinterpret_cast<char*>(descrSigmas[b]), Msize*sizeof(float));
      }
      else {
        for(int i=0; i<Msize; i++) descrSigmas[b][i] = sigma;
      }
      if(weights_file.eof()){
        std::cout << "Weights file " << wfile_string << " is too small." << std::endl;
        return -3;
      }
    }
    // Read network
    weights = new float*[Nlayers];
    biases = new float*[Nlayers];
    for (int l=0; l<Nlayers;l++){
      // Read weights matrix
      int Msize = Nspecies*arch[l]*arch[l+1];
      weights[l] = new float[Msize];
      weights_file.read(reinterpret_cast<char*>(weights[l]), Msize*sizeof(float));
      // Read biases
      Msize = Nspecies*arch[l+1];
      biases[l] = new float[Msize];
      weights_file.read(reinterpret_cast<char*>(biases[l]), Msize*sizeof(float));
      if(weights_file.eof()){
        std::cout << "Weights file " << wfile_string << " is too small." << std::endl;
        return -3;
      }
    }
    // Check if we're not at the end
    std::ifstream::pos_type fpos = weights_file.tellg();
    weights_file.seekg(0, std::ios::end);
    std::ifstream::pos_type epos = weights_file.tellg();
    if(fpos!=epos){
      std::cout << "Weights file " << wfile_string << " is too big." << std::endl;
      return -3;
    }
    weights_file.close();
  }

  return(0);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLatte::coeff(int narg, char **arg)
{
  if (!allocated) {
    allocate();
  }

  // We now expect a directory and the parameters file name (inside the directory) with all params
  if (narg != 3) {
    error->all(FLERR,"Format of pair_coeff command is\npair_coeff * *  network_directory\n");
  }

  std::cout << "Loading PANNA pair parameters from " << arg[2] << std::endl;
  int gpout = get_parameters(arg[2]);
  if(gpout==0){
    std::cout << "Network loaded!" << std::endl;
  }
  else{
    std::cout << "Error " << gpout << " while loading network!" << std::endl;
    exit(1);
  }

  for (int i=1; i<=atom->ntypes; i++) {
    for (int j=1; j<=atom->ntypes; j++) {
      cutsq[i][j] = rcut*rcut;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLatte::init_style()
{
  // request FULL neighbor list
  int list_style = NeighConst::REQ_FULL;
  neighbor->add_request(this, list_style);
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style LATTE requires newton pair on");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLatte::init_one(int i, int j)
{
  return sqrt(cutsq[i][j]);
}
