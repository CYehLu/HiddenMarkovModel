double runif(double, double);
int* sample(int, double *, int);

double addProbLog(double, double);
double mulProbLog(double, double);
double divProbLog(double, double);

void getMaxArgmax(int, double [], double *, int *);

double** calcAlpha(int, int, int, int *, double **, double **, double *);
double** calcBeta(int, int, int, int *, double **, double **);

void doEStep(int, int, int, int *, double **, double ***, double **, double **, double *);
void doMStep(int, int, int, int *, double **, double ***, double **, double **, double *);
    