// gcc -c hmm.c helper.o -std=c99 -o hmm.o -lm

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "helper.h"    // addProbLog, mulProbLog, getMaxArgmax, sample
#include "types.h"     // struct BestSeq, struct HMM


int *simulate(int Nstate, int Nseq, int Nobs, double **A, double **B, double *iniProb) {
    /*
    Given HMM {A (transition matrix), B (emission matrix), iniProb (initial probability)}, 
    simulate the sequence of observations generated by HMM.
    All of the probabilities are not in log scale (differ from other functions).
    */
    
    srand((unsigned)time(NULL));    // initialize `runif()`
    
    int *seqObs = malloc(Nseq * sizeof(int));
    int *seqState = malloc(Nseq * sizeof(int));
    
    int t = 0;
    int currState = sample(Nstate-1, iniProb, 1)[0];
    int currObs = sample(Nobs-1, B[currState], 1)[0];
    seqState[t] = currState;
    seqObs[t] = currObs;
    
    for (t = 1; t < Nseq; t++) {
        currState = sample(Nstate-1, A[currState], 1)[0];
        currObs = sample(Nobs-1, B[currState], 1)[0];
        seqState[t] = currState;
        seqObs[t] = currObs;
    }
    
    return seqObs;
}


double likelihood(int Nstate, int Nseq, int Nobs, int *seqObs, double **A, double **B, double *iniProb) {
    /* 
    Using forward algorithm to solve likelihood problem
    */
    
    // initialization & recursion step: compute `alpha[Nseq][Nstate]`
    double **alpha = calcAlpha(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    
    // termination step
    double res = alpha[Nseq-1][0];
    for (int s = 1; s < Nstate; s++) {
        res = addProbLog(res, alpha[Nseq-1][s]);
    }
    
    return res;
}


BestSeq decoding(int Nstate, int Nseq, int Nobs, int *seqObs, double **A, double **B, double *iniProb) {
    /* 
    Using Viterbi algorithm to solve decoding problem 
    */
    
    double v[Nseq][Nstate];
    int backptr[Nseq][Nstate];
    
    // initialization step
    for (int s = 0; s < Nstate; s++) {
        v[0][s] = mulProbLog(iniProb[s], B[s][seqObs[0]]);
        backptr[0][s] = 0;
    }
    
    // recursion step
    for (int t = 1; t < Nseq; t++) {
        for (int s = 0; s < Nstate; s++) {
            double candidateProb[Nstate];
            for (int s2 = 0; s2 < Nstate; s2++) {
                double tmp = mulProbLog(v[t-1][s2], A[s2][s]);
                candidateProb[s2] = mulProbLog(tmp, B[s][seqObs[t]]);
            }
            getMaxArgmax(Nstate, candidateProb, &(v[t][s]), &(backptr[t][s]));
        }
    }
    
    // termination step    
    double bestProb;
    int *bestSeq = malloc(Nseq * sizeof(int));
    
    getMaxArgmax(Nstate, v[Nseq-1], &bestProb, &(bestSeq[Nseq-1]));
    
    for (int t = Nseq-2; t >= 0; t--) {
        bestSeq[t] = backptr[t+1][bestSeq[t+1]];
    }
      
    BestSeq res = {bestProb, bestSeq};
    return res;
    
}


HMM learning(int Nstate, int Nseq, int Nobs, int *seqObs, HMM firstGuess, int maxiter, double tol) {
    // initial value
    double *iniProb = firstGuess.iniProb;
    double **A = firstGuess.A;
    double **B = firstGuess.B;
    
    // allocate `gamma` and `xi`
    double **gamma = malloc(Nseq * sizeof(double*));     // gamma[Nseq][Nstate]
    for (int t = 0; t < Nseq; t++) {
        gamma[t] = malloc(Nstate * sizeof(double));
    }
    
    double ***xi = malloc((Nseq-1) * sizeof(double**));   // xi[Nseq-1][Nstate][Nstate]
    for (int t = 0; t < Nseq-1; t++) {
        xi[t] = malloc(Nstate * sizeof(double*));
        for (int i = 0; i < Nstate; i++) {
            xi[t][i] = malloc(Nstate * sizeof(double));
        }
    }
    
    double prevScore = likelihood(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    
    // iterations
    for (int iter = 0; iter < maxiter; iter++) {
        printf("\n\n ********** iter = %d **********\n", iter);
        
        printf("parameters BEFORE updating:\n");
        for (int i = 0; i < Nstate; i++) {
            for (int j = 0; j < Nstate; j++) {
                printf("  A[%d][%d] = %lf  ", i, j, exp(A[i][j]));
            }
            printf("\n");
        }
        for (int i = 0; i < Nstate; i++) {
            for (int v = 0; v < Nobs; v++) {
                printf("  B[%d][%d] = %lf  ", i, v, exp(B[i][v]));
            }
            printf("\n");
        }
        for (int i = 0; i < Nstate; i++) {
            printf("  iniProb[%d] = %lf  ", i, exp(iniProb[i]));
        }
        printf("\n");
        
        // E-step and M-step
        doEStep(Nstate, Nseq, Nobs, seqObs, gamma, xi, A, B, iniProb);    // update `gamma` and `xi`
        doMStep(Nstate, Nseq, Nobs, seqObs, gamma, xi, A, B, iniProb);    // update `A`, `B` and `iniProb`
        
        printf("\n");
        printf("parameters AFTER updating:\n");
        for (int i = 0; i < Nstate; i++) {
            for (int j = 0; j < Nstate; j++) {
                printf("  A[%d][%d] = %lf  ", i, j, exp(A[i][j]));
            }
            printf("\n");
        }
        for (int i = 0; i < Nstate; i++) {
            for (int v = 0; v < Nobs; v++) {
                printf("  B[%d][%d] = %lf  ", i, v, exp(B[i][v]));
            }
            printf("\n");
        }
        for (int i = 0; i < Nstate; i++) {
            printf("  iniProb[%d] = %lf  ", i, exp(iniProb[i]));
        }
        printf("\n\n");
        
        // check convergence       
        double nowScore = likelihood(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
        double diff = exp(prevScore) - exp(nowScore);
        printf("check convergence:\n");
        printf("  nowScore = %lf, prevScore = %lf, diff = %lf, tol = %lf\n", exp(nowScore), exp(prevScore), diff, tol);
        printf("********** --------- **********\n\n");
        if (diff <= tol) 
            break;
            //printf("");
        else
            prevScore = nowScore;
    }
    
    // free memory and return
    for (int t = 0; t < Nseq; t++) {
        free(gamma[t]);
    }
    free(gamma);
    
    for (int t = 0; t < Nseq-1; t++) {
        for (int i = 0; i < Nstate; i++) {
            free(xi[t][i]);
        }
    }
    free(xi);
    
    printf("\n\n\n");
    
    HMM result = {iniProb, A, B};
    return result;
}