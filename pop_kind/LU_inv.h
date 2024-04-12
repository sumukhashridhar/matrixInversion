#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void lu_decomp(double *, double *, double *, int);
void forward_sub(double *, double *, double *, int);
void backward_sub(double *, double *, double *, int);
void lu_invert_matrix(double *, double *, int);
