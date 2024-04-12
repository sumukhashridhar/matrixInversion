#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// declare all memory allocation in a struct
typedef struct {
    double* A;
    double* L;
    double* U;
    double* y;
    double *e;
    double* A_inv;
    double* Id;
} dataStore;

dataStore data;

// LU functions
void lu_decomp(int);
void forward_sub(double *, int);
void backward_sub(double *, int);
void lu_invert_matrix(int);

// helper functions
void assign_memory(int);
void fill_matrices(int);
void free_memory();
