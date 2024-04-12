#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// declare pointers for CUDA
typedef struct {
    double* A;
    double* L;
    double* U;
    double* y;
    double *e;
    double* A_inv;
    double* Id;
} cudaMem;

