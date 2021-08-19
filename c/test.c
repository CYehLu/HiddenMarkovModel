#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void func(int, int, double [][2]);
void func2(int, int, double**);

int main(void) {
    int NROW = 2;
    int NCOL = 2;
    double a[NROW][NCOL];
    
    for (int i = 0; i < NROW; i++)
        for (int j = 0; j < NCOL; j++)
            a[i][j] = (double)(i+j);
    
    func(NROW, NCOL, a);
    
    double** b;
    b = malloc(NROW * sizeof(int*));
    for (int i = 0; i < NROW; i++) {
        b[i] = malloc(NCOL * sizeof(int));
        
        for (int j = 0; j < NCOL; j++) {
            b[i][j] = (double)(i + j);
        }
    }
    
    func2(NROW, NCOL, b);
    
    for (int i = 0; i < NROW; i++) {
        free(b[i]);
    }
    free(b);
    
    func2(NROW, NCOL, &(a[0]));
    
    return 0;
}

void func(int m, int n, double a[][n]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
}


void func2(int m, int n, double** a) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
}

