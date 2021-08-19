// gcc main_likelihood.c hmm.o -std=c99 -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "hmm.h"

#define Nstate 2
#define Nobs 3
#define Nseq 3

int main(void) {   
    // observation sequence
    int* seqObs;
    seqObs = malloc(Nseq * sizeof(int));
    seqObs[0] = 2;
    seqObs[1] = 0;
    seqObs[2] = 2;
    
    // initial probability
    double* iniProb;
    iniProb = malloc(Nstate * sizeof(double));
    iniProb[0] = log(0.8);
    iniProb[1] = log(0.2);
    
    // transition matrix
    double** A;
    A = malloc(Nstate * sizeof(double*));
    for (int t = 0; t < Nstate; t++) {
        A[t] = malloc(Nstate * sizeof(double));
    }
    A[0][0] = log(0.6);  A[0][1] = log(0.4);
    A[1][0] = log(0.5);  A[1][1] = log(0.5);
    
    // emission matrix
    double** B;
    B = malloc(Nstate * sizeof(double*));
    for (int t = 0; t < Nstate; t++) {
        B[t] = malloc(Nobs * sizeof(double));
    }
    B[0][0] = log(0.2);  B[0][1] = log(0.4);  B[0][2] = log(0.4);
    B[1][0] = log(0.5);  B[1][1] = log(0.4);  B[1][2] = log(0.1);
    
    // HMM: likelihood problem
    double res = likelihood(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    printf("res = %f\n", res);
    
    return 0;
}
