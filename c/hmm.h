#include "types.h"    // struct BestSeq, struct HMM

int* simulate(int, int, int, double **, double **, double *);
double likelihood(int, int, int, int *, double **, double **, double *);
BestSeq decoding(int, int , int , int *, double **, double **, double *);
HMM learning(int, int, int, int *, HMM, int, double);