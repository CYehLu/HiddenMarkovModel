// gcc main_simulate.c hmm.o -std=c99 -o simulate

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "hmm.h"

#define Nstate 2
#define Nobs 3
#define Nseq 30


double** malloc2dArray(int, int);

int main(void) {   
    // initial probability
    double iniProb[Nstate] = {0.8, 0.2};
    
    // transition matrix
    double **A = malloc2dArray(Nstate, Nstate);
    A[0][0] = 0.6;  A[0][1] = 0.4;
    A[1][0] = 0.5;  A[1][1] = 0.5;
    
    // emission matrix
    double **B = malloc2dArray(Nstate, Nobs);
    B[0][0] = 0.2;  B[0][1] = 0.4;  B[0][2] = 0.4;
    B[1][0] = 0.5;  B[1][1] = 0.4;  B[1][2] = 0.1;
    
    // simulate HMM
    int *seqObs = simulate(Nstate, Nseq, Nobs, A, B, iniProb);
    
    printf("Simulated sequence of observations:\n");
    for (int i = 0; i < Nseq; i++) {
        if (i%20 == 0) printf("\n");
        printf("%d  ", seqObs[i]);
    }
    printf("\n\n");
    
    return 0;
}


double** malloc2dArray(int ncol, int nrow) {
    double** ptr = malloc(nrow * sizeof(double*));
    for (int i = 0; i < nrow; i++) {
        ptr[i] = malloc(ncol * sizeof(double));
    }
    return ptr;
}
