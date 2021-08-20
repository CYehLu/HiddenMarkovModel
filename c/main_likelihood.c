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
    int seqObs[Nseq] = {2, 0, 2};
    
    // initial probability
    double iniProb[Nstate] = {log(0.8), log(0.2)};
    
    // transition matrix
    double A[Nstate][Nstate] = {
        {log(0.6), log(0.4)},
        {log(0.5), log(0.5)}
    };
    
    // emission matrix
    double B[Nstate][Nobs] = {
        {log(0.2), log(0.4), log(0.4)},
        {log(0.5), log(0.4), log(0.1)}
    };
    
    // HMM: likelihood problem
    double res = likelihood(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    printf("res = %f\n", res);
    
    return 0;
}
