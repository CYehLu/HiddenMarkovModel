// gcc -c helper.c -std=c99 -o helper.o -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


double runif(double min, double max) {
    /* 
    generate random number from uniform distribution: U(min, max)
    initialization `srand((unsigned)time(NULL));` should be added before using it
    */
    double x = (max-min) * (double)rand() / (double)(RAND_MAX) + min;
    return x;
}


int* sample(int n, double *prob, int size) {
    /* 
    generate weighted random sample from {0, 1, ..., n}.
    */
    
    double cumprob[n+1];
    cumprob[0] = prob[0];
    for (int i = 1; i <= n; i++) {
        cumprob[i] = cumprob[i-1] + prob[i];
    }
    
    int *res = malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        double x = runif(0.0, 1.0);
        
        if (x < cumprob[0]) 
            res[i] = 0;
        
        for (int j = 1; j < n+1; j++) {
            if ((x >= cumprob[j-1]) && (x < cumprob[j])) {res[i] = j; break;}
        }
    }
    
    return res;
}


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


double divProbLog(double logProb1, double logProb2) {
    /*
    result = log(Prob1/Prob2) = log(Prob1) - log(Prob2)
    */
    return logProb1 - logProb2;
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


double** calcAlpha(int Nstate, int Nseq, int Nobs, int *seqObs, double **A, double **B, double *iniProb) {
    /* 
    Calculate forward probability 
    */
    
    // allocate alpha[Nseq][Nstate] as 0
    double **alpha = malloc(Nseq * sizeof(double*));   
    for (int t = 0; t < Nseq; t++) {
        alpha[t] = calloc(Nstate, sizeof(double));
    }
    
    // initialization step
    for (int s = 0; s < Nstate; s++) 
        alpha[0][s] = mulProbLog(iniProb[s], B[s][seqObs[0]]);
    
    // recursion step   
    for (int t = 1; t < Nseq; t++) {
        for (int s = 0; s < Nstate; s++) {
            
            // alpha[t][s] = sum_s2(alpha[t-1][s2] * A[s2][s] * B[s][seqObs[t]])
            alpha[t][s] = mulProbLog(alpha[t-1][0], A[0][s]);     
            for (int s2 = 1; s2 < Nstate; s2++) {
                double tmp = mulProbLog(alpha[t-1][s2], A[s2][s]);
                alpha[t][s] = addProbLog(alpha[t][s], tmp);
            }
            alpha[t][s] = mulProbLog(alpha[t][s], B[s][seqObs[t]]);
        }
    }
    
    return alpha;
}


double** calcBeta(int Nstate, int Nseq, int Nobs, int *seqObs, double **A, double **B) {
    /* 
    Calculate backward probability 
    */
    
    // allocate beta[Nseq][Nstate]
    double **beta = malloc(Nseq * sizeof(double*));
    for (int t = 0; t < Nseq; t++) {
        beta[t] = malloc(Nstate * sizeof(double));
    }
    
    // initialization step
    for (int s = 0; s < Nstate; s++) {
        beta[Nseq-1][s] = log(1.);
    }
    
    // recursion step
    for (int t = Nseq-2; t >= 0; t--) {
        for (int s = 0; s < Nstate; s++) {
            
            // beta[t][s] = sum_s2(A[s][s2] * B[s2][seqObs[t+1]] * beta[t+1][s2]);
            double tmp = mulProbLog(A[s][0], B[0][seqObs[t+1]]);
            beta[t][s] = mulProbLog(tmp, beta[t+1][0]);
            for (int s2 = 1; s2 < Nstate; s2++) {
                tmp = mulProbLog(A[s][s2], B[s2][seqObs[t+1]]);
                tmp = mulProbLog(tmp, beta[t+1][s2]);
                beta[t][s] = addProbLog(beta[t][s], tmp);
            }
            
        }
    }
    
    return beta;
}


void doEStep(int Nstate, int Nseq, int Nobs, int *seqObs, double **gamma, double ***xi, double **A, double **B, double *iniProb) {
    /*
    Update `gamma` and `xi`
    */
    
    double **alpha = calcAlpha(Nstate, Nseq, Nobs, seqObs, A, B, iniProb);
    double **beta = calcBeta(Nstate, Nseq, Nobs, seqObs, A, B);
    
    // norm[t] = sum_j(alpha[t][j] * beta[t][j])
    double norm[Nseq];
    for (int t = 0; t < Nseq; t++) {
        norm[t] = mulProbLog(alpha[t][0], beta[t][0]);
        for (int j = 1; j < Nstate; j++) {
            double tmp = mulProbLog(alpha[t][j], beta[t][j]);
            norm[t] = addProbLog(norm[t], tmp);
        }
    }
    
    // update `gamma`: gamma[t][i] = alpha[t][i] * beta[t][i] / norm[t]
    for (int t = 0; t < Nseq; t++) {
        for (int i = 0; i < Nstate; i++) {
            double numerator = mulProbLog(alpha[t][i], beta[t][i]);
            gamma[t][i] = divProbLog(numerator, norm[t]);
        }
    }
    
    // update `xi`: xi[t][i][j] = alpha[t][i] * A[i][j] * B[j][seqObs[t+1]] * beta[t+1][j] / norm[t]
    for (int t = 0; t < Nseq-1; t++) {
        for (int i = 0; i < Nstate; i++) {
            for (int j = 0; j < Nstate; j++) {
                double term1 = mulProbLog(alpha[t][i], A[i][j]);
                double term2 = mulProbLog(B[j][seqObs[t+1]], beta[t+1][j]);
                double numerator = mulProbLog(term1, term2);
                xi[t][i][j] = divProbLog(numerator, norm[t]);
            }
        }
    }
    
    // free memory
    for (int i = 0; i < Nstate; i++) {
        free(alpha[i]);
        free(beta[i]);
    }
    free(alpha);
    free(beta);
}


void doMStep(int Nstate, int Nseq, int Nobs, int *seqObs, double **gamma, double ***xi, double **A, double **B, double *iniProb) {
    /*
    Update `iniProb`, `A` and `B`
    */
    
    // 1. update `A`
    
    double xiSumT[Nstate][Nstate];
    double xiSumTJ[Nstate];
    
    // initialize `xiSumT` and `xiSumTJ` as 0.
    for (int i = 0; i < Nstate; i++) {
        xiSumTJ[i] = log(0.);
        for (int j = 0; j < Nstate; j++) {
            xiSumT[i][j] = log(0.);
        }
    }
    
    // xiSumT[i][j] = sum_t(xi[t][i][j])
    for (int t = 0; t < Nseq-1; t++) {
        for (int i = 0; i < Nstate; i++) {
            for (int j = 0; j < Nstate; j++) {
                xiSumT[i][j] = addProbLog(xiSumT[i][j], xi[t][i][j]);
            }
        }
    }
    
    // xiSumTJ[i] = sum_t(sum_j(xt[t][i][j]))
    for (int t = 0; t < Nseq-1; t++) {
        for (int i = 0; i < Nstate; i++) {
            for (int j = 0; j < Nstate; j++) {
                xiSumTJ[i] = addProbLog(xiSumTJ[i], xi[t][i][j]);
            }
        }
    }
    
    // A[i][j] = xiSumT[i][j] / xiSumTJ[i]
    for (int i = 0; i < Nstate; i++) {
        for (int j = 0; j < Nstate; j++) {
            A[i][j] = divProbLog(xiSumT[i][j], xiSumTJ[i]);
        }
    }
    
    // 2. update `B`
    
    double numerator[Nstate][Nobs];
    double denominator[Nstate];
    
    // initialize `numerator` and `denominator`
    for (int j = 0; j < Nstate; j++) {
        denominator[j] = log(0.);
        for (int v = 0; v < Nobs; v++) {
            numerator[j][v] = log(0.);
        }
    }
    
    // numerator[j][v] = sum_t(gamma[t][j]) s.t. seqObs[t] == v
    for (int t = 0; t < Nseq; t++) {
        for (int j = 0; j < Nstate; j++) {
            for (int v = 0; v < Nobs; v++) {
                if (seqObs[t] == v)
                    numerator[j][v] = addProbLog(numerator[j][v], gamma[t][j]);
            }
        }
    }
    
    // denominator[j] = sum_t(gamma[t][j])
    for (int t = 0; t < Nseq; t++) {
        for (int j = 0; j < Nstate; j++) {
            denominator[j] = addProbLog(denominator[j], gamma[t][j]);
        }
    }
    
    // B[j][v] = numerator[j][v] / denominator[j]
    for (int j = 0; j < Nstate; j++) {
        for (int v = 0; v < Nobs; v++) {
            B[j][v] = divProbLog(numerator[j][v], denominator[j]);
        }
    }
    
    // 3. update `iniProb`
    for (int j = 0; j < Nstate; j++) {
        iniProb[j] = gamma[0][j];
    }
}

