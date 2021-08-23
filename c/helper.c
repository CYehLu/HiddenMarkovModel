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

