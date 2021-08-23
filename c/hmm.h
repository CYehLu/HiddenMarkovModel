#include "types.h"    // struct BestSeq

int* simulate(int, int, int, double **, double **, double *);
double likelihood(int, int, int, int *, double **, double **, double *);
BestSeq decoding(int, int , int , int *, double **, double **, double *);