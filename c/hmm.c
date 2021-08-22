// gcc -c hmm.c -std=c99 -o hmm.o -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "types.h"    // struct BestSeq


double addProbLog(double logProb1, double logProb2) {
    if (logProb2 > logProb1) {
        double tmp = logProb1;
        logProb1 = logProb2;
        logProb2 = tmp;
    }
    return logProb1 + log1p(exp(logProb2 - logProb1));
}


double mulProbLog(double logProb1, double logProb2) {
    return logProb1 + logProb2;
}


void getMaxArgmax(int n, double list[], double *max, int *argmax) {   
    *max = list[0];
    *argmax = 0;
    
    for (int i = 1; i < n; i++) {
        if (list[i] > *max) {
            *max = list[i];
            *argmax = i;
        }
    }
}


double likelihood(int Nstate, int Nseq, int Nobs, int *seqObs, double **A, double **B, double *iniProb) {
    // Using forward algorithm to solve likelihood problem
    
    double alpha[Nstate][Nseq]; 
    
    // initial step
    for (int s = 0; s < Nstate; s++) 
        alpha[s][0] = mulProbLog(iniProb[s], B[s][seqObs[0]]);
    
    // recursion step   
    for (int t = 1; t < Nseq; t++) {
        for (int s = 0; s < Nstate; s++) {
            alpha[s][t] = mulProbLog(alpha[0][t-1], A[0][s]);     
            
            for (int s2 = 1; s2 < Nstate; s2++) {
                double tmp = mulProbLog(alpha[s2][t-1], A[s2][s]);
                alpha[s][t] = addProbLog(alpha[s][t], tmp);
            }
            
            alpha[s][t] = mulProbLog(alpha[s][t], B[s][seqObs[t]]);
        }
    }
    
    // termination step
    double res = alpha[0][Nseq-1];
    for (int s = 1; s < Nstate; s++) {
        res = addProbLog(res, alpha[s][Nseq-1]);
    }
    
    return res;
}


BestSeq decoding(int Nstate, int Nseq, int Nobs, int *seqObs, double **A, double **B, double *iniProb) {
    // Using Viterbi algorithm to solve decoding problem
    
    double v[Nstate][Nseq];
    int backptr[Nstate][Nseq];  
    
    // initialization step
    for (int s = 0; s < Nstate; s++) {
        v[s][0] = mulProbLog(iniProb[s], B[s][seqObs[0]]);
        backptr[s][0] = 0;
    }
    
    // recursion step
    for (int t = 1; t < Nseq; t++) {
        for (int s = 0; s < Nstate; s++) {
            double candidateProb[Nstate];
            for (int s2 = 0; s2 < Nstate; s2++) {
                double tmp = mulProbLog(v[s2][t-1], A[s2][s]);
                candidateProb[s2] = mulProbLog(tmp, B[s][seqObs[t]]);
            }
            getMaxArgmax(Nstate, candidateProb, &(v[s][t]), &(backptr[s][t]));
        }
    }
    
    // termination step    
    int *bestSeq = malloc(Nseq * sizeof(int));
    bestSeq[Nseq-1] = 0;
    double bestProb = v[0][Nseq-1];
    
    for (int s = 1; s < Nstate; s++) {
        if (v[s][Nseq-1] > bestProb) {
            bestProb = v[s][Nseq-1];
            bestSeq[Nseq-1] = s;
        }
    }
    
    for (int t = Nseq-2; t >= 0; t--) {
        bestSeq[t] = backptr[bestSeq[t+1]][t+1];
    }
      
    BestSeq res = {bestProb, bestSeq};
    return res;
}
