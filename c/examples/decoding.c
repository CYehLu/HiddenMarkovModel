// gcc decoding.c ../src/hmm.o ../src/helper.o -I../src/ -std=c99 -lm -o decoding

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "hmm.h"

#define Nstate 2
#define Nobs 3
#define Nseq 3


double** malloc2dArray(int, int);

int main(void) {
    /* 
    Case 1 
    */
    
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
    
    // HMM: decoding problem
    BestSeq res = decoding(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    
    // print result
    printf("Case 1\n");
    printf("max log-probability = %lf\n", res.bestProb);
    printf("corresponding hidden state sequence:\n");
    for (int t = 0; t < Nseq; t++) {
        printf(" %d", res.bestSeq[t]);
    }
    printf("\n");
    
    
    /* 
    Case 2 
    https://en.wikipedia.org/wiki/Viterbi_algorithm
    */
    
    // observation sequence
    seqObs[0] = 0; seqObs[1] = 1; seqObs[2] = 2;
    
    // initial probability
    iniProb[0] = log(0.6); iniProb[1] = log(0.4);
    
    // transition matrix
    A[0][0] = log(0.7);  A[0][1] = log(0.3);
    A[1][0] = log(0.4);  A[1][1] = log(0.6);
    
    // emission matrix
    B[0][0] = log(0.5);  B[0][1] = log(0.4);  B[0][2] = log(0.1);
    B[1][0] = log(0.1);  B[1][1] = log(0.3);  B[1][2] = log(0.6);
    
    // HMM: decoding problem
    res = decoding(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    
    // print result
    printf("\nCase 2\n");
    printf("max log-probability = %lf\n", res.bestProb);
    printf("corresponding hidden state sequence:\n");
    for (int t = 0; t < Nseq; t++) {
        printf(" %d", res.bestSeq[t]);
    }
    printf("\n");
    
    
    /* 
    Free memory 
    */
    for (int i = 0; i < Nstate; i++) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
    
    return 0;
}


double** malloc2dArray(int ncol, int nrow) {
    double** ptr = malloc(nrow * sizeof(double*));
    for (int i = 0; i < nrow; i++) {
        ptr[i] = malloc(ncol * sizeof(double));
    }
    return ptr;
}

