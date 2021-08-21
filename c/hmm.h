#include "types.h"    // struct BestSeq

double addProbLog(double, double);
double mulProbLog(double, double);
double likelihood(int, int, int, int *, double **, double **, double *);
BestSeq decoding(int, int , int , int *, double **, double **, double *);