// gcc -c hmm.c -std=c99 -o hmm.o -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


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


double likelihood(int Nstate, int Nseq, int Nobs, int *seqObs, double (*A)[Nstate], double (*B)[Nobs], double *iniProb) {
    double alpha[Nstate][Nseq];
    
    // initial step
    for (int s = 0; s < Nstate; s++) 
        alpha[s][0] = mulProbLog(iniProb[s], B[s][seqObs[0]]);
    
    // recursion step
    double tmp;
    
    for (int t = 1; t < Nseq; t++) {
        for (int s = 0; s < Nstate; s++) {
            alpha[s][t] = mulProbLog(alpha[0][t-1], A[0][s]);     
            
            for (int s2 = 1; s2 < Nstate; s2++) {
                tmp = mulProbLog(alpha[s2][t-1], A[s2][s]);
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
