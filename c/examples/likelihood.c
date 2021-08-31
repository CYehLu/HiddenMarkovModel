// gcc likelihood.c ../src/hmm.o ../src/helper.o -I../src/ -std=c99 -lm -o likelihood

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "hmm.h"

#define Nstate 2
#define Nobs 3
#define Nseq 3


double** malloc2dArray(int, int);

int main(void) {   
    // observation sequence
    int seqObs[Nseq] = {2, 0, 2};
    
    // initial probability
    double iniProb[Nstate] = {log(0.8), log(0.2)};
    
    // transition matrix
    double **A = malloc2dArray(Nstate, Nstate);
    A[0][0] = log(0.6);  A[0][1] = log(0.4);
    A[1][0] = log(0.5);  A[1][1] = log(0.5);
    
    
    // emission matrix
    double **B = malloc2dArray(Nstate, Nobs);
    B[0][0] = log(0.2);  B[0][1] = log(0.4);  B[0][2] = log(0.4);
    B[1][0] = log(0.5);  B[1][1] = log(0.4);  B[1][2] = log(0.1);
    
    // HMM: likelihood problem
    double res = likelihood(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    printf("log-likelihood = %f\n", res);
    
    return 0;
}


double** malloc2dArray(int ncol, int nrow) {
    double** ptr = malloc(nrow * sizeof(double*));
    for (int i = 0; i < nrow; i++) {
        ptr[i] = malloc(ncol * sizeof(double));
    }
    return ptr;
}

