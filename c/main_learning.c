// gcc main_learning.c hmm.o -std=c99 -o learning

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "hmm.h"

#define Nstate 2
#define Nobs 3
#define Nseq 10000


double** malloc2dArray(int, int);
void showParams(int, int, HMM, HMM);

int main(void) {   
    
    /*
    Define the true HMM: {iniProb, A, B} and simulate observations
    (probabilities are NOT in log scale)
    */
    
    double iniProb[Nstate] = {0.8, 0.2};
    
    double **A = malloc2dArray(Nstate, Nstate);
    A[0][0] = 0.65;  A[0][1] = 0.35;
    A[1][0] = 0.25;  A[1][1] = 0.75;
    
    double **B = malloc2dArray(Nstate, Nobs);
    B[0][0] = 0.2;  B[0][1] = 0.4;  B[0][2] = 0.4;
    B[1][0] = 0.5;  B[1][1] = 0.4;  B[1][2] = 0.1;
    
    int *seqObs = simulate(Nstate, Nseq, Nobs, A, B, iniProb);
    
    
    /* 
    Learn the parameters {estIniProb, estA, estB} through the given observations
    */
    
    // define first guess parameters (probabilities are in log scale)
    int seed = (unsigned)time(NULL);
    HMM firstGuess = randomInitParams(Nstate, Nseq, Nobs, seed);
    
    printf("Initial parameters\n");
    printf("------------------\n");
    HMM trueHMM = {iniProb, A, B};
    showParams(Nstate, Nobs, firstGuess, trueHMM);
    
    // iteration settings
    int maxiter = 100;       // maximum number of iteration
    double tol = 1e-4;       // stop iteration if log-likelihood increment is lower than `tol`
    int verbose = 1;         // display iteration history
    
    // start learning
    HMM estHMM = learning(Nstate, Nseq, Nobs, seqObs, firstGuess, maxiter, tol, verbose);
    
    
    /*
    Compare the estimated parameters and the true parameters
    */
    printf("Parameter = estimated value (true value)\n");
    printf("----------------------------------------\n");
    showParams(Nstate, Nobs, estHMM, trueHMM);
    
    return 0;
}


double** malloc2dArray(int ncol, int nrow) {
    double** ptr = malloc(nrow * sizeof(double*));
    for (int i = 0; i < nrow; i++) {
        ptr[i] = malloc(ncol * sizeof(double));
    }
    return ptr;
}


void showParams(int Nst, int Nob, HMM estParams, HMM trueParams) {
    for (int i = 0; i < Nst; i++) {
        for (int j = 0; j < Nst; j++) {
            printf("  A[%d][%d] = %lf (%lf)  ", i, j, exp(estParams.A[i][j]), trueParams.A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    for (int i = 0; i < Nst; i++) {
        for (int v = 0; v < Nob; v++) {
            printf("  B[%d][%d] = %lf (%lf)  ", i, v, exp(estParams.B[i][v]), trueParams.B[i][v]);
        }
        printf("\n");
    }
    printf("\n");
    
    for (int i = 0; i < Nst; i++) {
        printf("  iniProb[%d] = %lf (%lf)  ", i, exp(estParams.iniProb[i]), trueParams.iniProb[i]);
    }
    printf("\n");
}

