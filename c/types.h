typedef struct BestSeq {
    double bestProb;
    int *bestSeq;
} BestSeq;


typedef struct HMM {
    double *iniProb;   // initial probability
    double **A;        // transition matrix
    double **B;        // emission matrix
} HMM;