#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void lu_decomposition(double *, double *, double *, int);
void forward_substitution(double *, double *, double *, int);
void backward_substitution(double *, double *, double *, int);
void lu_invert_matrix(double *, double *, int);
