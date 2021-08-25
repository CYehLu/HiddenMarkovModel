// gcc main_learning.c hmm.o -std=c99 -o learning

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "hmm.h"

#define Nstate 2
#define Nobs 3
#define Nseq 100


double** malloc2dArray(int, int);

int main(void) {   
    
    /*
    Define the true HMM: {iniProb, A, B} and simulate observations
    (probabilities are NOT in log scale)
    */
    double iniProb[Nstate] = {0.8, 0.2};
    
    double **A = malloc2dArray(Nstate, Nstate);
    A[0][0] = 0.6;  A[0][1] = 0.4;
    A[1][0] = 0.5;  A[1][1] = 0.5;
    
    double **B = malloc2dArray(Nstate, Nobs);
    B[0][0] = 0.2;  B[0][1] = 0.4;  B[0][2] = 0.4;
    B[1][0] = 0.5;  B[1][1] = 0.4;  B[1][2] = 0.1;
    
    int *seqObs = simulate(Nstate, Nseq, Nobs, A, B, iniProb);
    
    
    /* 
    Learn the parameters {estIniProb, estA, estB} through the given observations
    */
    
    // define first guess parameters (probabilities are in log scale)
    double firstGeussIniProb[Nstate] = {log(0.5), log(0.5)};
    
    double **firstGuessA = malloc2dArray(Nstate, Nstate);
    for (int i = 0; i < Nstate; i++) {
        for (int j = 0; j < Nstate; j++) {
            firstGuessA[i][j] = log(1./Nstate);
        }
    }
    
    double **firstGuessB = malloc2dArray(Nstate, Nobs);
    for (int i = 0; i < Nstate; i++) {
        for (int v = 0; v < Nobs; v++) {
            firstGuessB[i][v] = log(1./Nobs);
        }
    }
    
    HMM firstGuess = {firstGeussIniProb, firstGuessA, firstGuessB};
    
    // iteration settings
    int maxiter = 100;
    double tol = 1e-5;
    
    // start learning
    HMM estHMM = learning(Nstate, Nseq, Nobs, seqObs, firstGuess, maxiter, tol);
    
    
    /*
    Compare the estimated parameters and the true parameters
    */
    printf("Parameter = estimated value (true value)\n");
    printf("----------------------------------------\n");
    
    for (int i = 0; i < Nstate; i++) {
        for (int j = 0; j < Nstate; j++) {
            printf("  A[%d][%d] = %lf (%lf)  ", i, j, exp(estHMM.A[i][j]), A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    for (int i = 0; i < Nstate; i++) {
        for (int v = 0; v < Nobs; v++) {
            printf("  B[%d][%d] = %lf (%lf)  ", i, v, exp(estHMM.B[i][v]), B[i][v]);
        }
        printf("\n");
    }
    printf("\n");
    
    for (int i = 0; i < Nstate; i++) {
        printf("  iniProb[%d] = %ld (%lf)  ", i, exp(estHMM.iniProb[i]), iniProb[i]);
    }
    printf("\n");
    
    return 0;
}


double** malloc2dArray(int ncol, int nrow) {
    double** ptr = malloc(nrow * sizeof(double*));
    for (int i = 0; i < nrow; i++) {
        ptr[i] = malloc(ncol * sizeof(double));
    }
    return ptr;
}
