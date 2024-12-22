# matrixInversion

## Contents for my Masters thesis "Optimized Block-Level Matrix Inversion Kernels for Small, Batched Matrices on GPUs"

### Abstract

This thesis examines efficient matrix inversion techniques on NVIDIA GPUs, focusing on two primary methods: LU decomposition (LUD) for square matrices and singular value decomposition (SVD) for non-square matrices. We confine ourselves to small/medium sized matrices, execute them in batches of order 1,00,000. We do this efficiently by doing the inversion of a matrix on a single thread block. Finally, we compare and analyze these methods, and benchmark them separately in cuBLAS. Our objective is to develop an efficient implementation of batched matrix inversion operations for integration into the SeisSol framework.

### Build Instructions

To compile the CUDA program with optimizations:

```bash

nvcc -O3 -DMATRIXSIZE=32 -DNUMMATRICES=1000 -DNUMTHREADS=32 -g --resource-usage --ptxas-options=-v --expt-relaxed-constexpr --extra-device-vectorization --use_fast_math --default-stream per-thread --std=c++17 --extended-lambda --expt-extended-lambda --Werror cross-execution-space-call --dlink-time-opt --display-error-number --generate-line-info --source-in-ptx -Xcompiler -ffast-math -Xcompiler -march=native -Xcompiler -funroll-loops -Xcompiler -fomit-frame-pointer -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xcompiler -fno-stack-protector -Xcompiler -fno-math-errno -Xptxas --opt-level=3 -Xptxas --allow-expensive-optimizations=true -Xptxas -dlcm=cg -Xptxas -dscm=wt -Xptxas --preserve-relocs --restrict -lineinfo -arch=sm_86 -t 0 luBatchedInplace.cu -o custom

```

To profile with NCU:

```bash

ncu --set=full --import-source yes --target-processes all --replay-mode kernel --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section SchedulerStats --section SourceCounters --section SpeedOfLight --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --clock-control base --launch-skip 0 --kernel-name-base mangled -f -o profile_output ./custom

```

To compile cuBLAS benchmark:

```bash

nvcc -O3 -DMATRIXSIZE=32 -DNUMMATRICES=1000 -DNUMTHREADS=32 -g --resource-usage --ptxas-options=-v --expt-relaxed-constexpr --extra-device-vectorization --use_fast_math --default-stream per-thread --std=c++17 --extended-lambda --expt-extended-lambda --Werror cross-execution-space-call --dlink-time-opt --display-error-number --generate-line-info --source-in-ptx -Xcompiler -ffast-math -Xcompiler -march=native -Xcompiler -funroll-loops -Xcompiler -fomit-frame-pointer -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xcompiler -fno-stack-protector -Xcompiler -fno-math-errno -Xptxas --opt-level=3 -Xptxas --allow-expensive-optimizations=true -Xptxas -dlcm=cg -Xptxas -dscm=wt -Xptxas --preserve-relocs --restrict -lineinfo -arch=sm_86 -t 0 luBatchedInplace.cu -o benchmark

```

To profile with NCU:

```bash

ncu --set=full --import-source yes --target-processes all --replay-mode kernel --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section SchedulerStats --section SourceCounters --section SpeedOfLight --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --clock-control base --launch-skip 0 --kernel-name-base mangled -f -o profile_output ./benchmark

```
