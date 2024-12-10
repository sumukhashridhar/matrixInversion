#include <stdio.h>
#include <cuda.h>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Reserved shared memory per block: %zu bytes\n", prop.reservedSharedMemPerBlock );
        printf("  Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlockOptin);
        printf("  Shared memory per multiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  Max block per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Total constant memory: %zu bytes\n", prop.totalConstMem);
    }
    return 0;
}
