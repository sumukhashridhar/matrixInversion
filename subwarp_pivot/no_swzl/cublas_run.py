import numpy as np
import subprocess
import os
import time
from typing import List, Tuple
import json
from datetime import datetime

def create_output_directories(timestamp: str) -> tuple:
    """Create directories for storing NCU profiles and runtime results."""
    base_dir = os.path.join("cublas_results", timestamp)
    cublas_dir = os.path.join(base_dir, "cublas_profiles")
    runtime_dir = os.path.join(base_dir, "cublas_runtime_results")
    
    # Create directories
    os.makedirs(cublas_dir, exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)
    
    return base_dir, cublas_dir, runtime_dir


def load_modules():
    """Load required modules using module load command"""
    modules = ["cuda-11.6.0", "openmpi-4.0.2", "binutils-2.41-static"]
    module_cmd = f"module load {' '.join(modules)}"

    subprocess.run(module_cmd, shell=True, check=True, executable='/bin/bash')
    print("Modules loaded successfully")
    print("Loaded modules are: ", modules)


def compile_cublas(matrix_size: int, num_matrices: int) -> None:
    """Compile CUDA program with given parameters."""
    compile_command = [
        "nvcc", "-O3", "-lcublas",
        f"-DINP_MATRIX_SIZE={matrix_size}",
        f"-DINP_BATCH_SIZE={num_matrices}",
        "-g", # Generate debug info
        "--resource-usage", # Show resource usage for better optimization
        "--ptxas-options=-v", # Show PTXAS optimization details
        "--expt-relaxed-constexpr", # Enable relaxed constexpr for better optimization
        "--extra-device-vectorization", # Enable additional device vectorization
        "--use_fast_math", # Enable fast math operations
        "--default-stream", "per-thread", # Use per-thread default stream for better performance
        "--std=c++17",  # Enable C++17 features for better template optimization
        "--extended-lambda",  # Enable extended lambda features
        "--expt-extended-lambda",  # Additional lambda optimizations
        "--Werror", "cross-execution-space-call",  # Strict checking for device/host function calls
        "--dlink-time-opt",  # Enable device-side link time optimization
        "--display-error-number",  # Show detailed error numbers for better debugging
        "--generate-line-info",  # More detailed line info for profiling
        "--source-in-ptx",  # Include source in PTX for better debugging
        # Host compiler optimizations
        "-Xcompiler", "-ffast-math", # Fast math operations
        "-Xcompiler", "-march=native", # Use native architecture for host
        "-Xcompiler", "-funroll-loops", # Unroll loops for better performance
        "-Xcompiler", "-fomit-frame-pointer",  # Remove frame pointer for better performance
        "-Xcompiler", "-ffunction-sections",  # Place each function in its own section
        "-Xcompiler", "-fdata-sections",  # Place each data item in its own section
        "-Xcompiler", "-fno-stack-protector",  # Disable stack protector
        "-Xcompiler", "-fno-math-errno",  # Don't set errno for math functions
        # PTX optimization flags
        "-Xptxas", "--opt-level=3", # Maximum optimization level
        "-Xptxas", "--allow-expensive-optimizations=true",
        "-Xptxas", "-dlcm=cg",  # Cache global memory operations
        "-Xptxas", "-dscm=wt",  # Write-through cache for shared memory
        # "-Xptxas", "--def-load-cache=ca",  # Cache load operations
        # "-Xptxas", "--def-store-cache=wb",  # Write-back cache for stores
        # "-Xptxas", "--unroll-count=256",  # Aggressive loop unrolling
        "-Xptxas", "--preserve-relocs",  # Preserve relocations for link-time opts
        # Register and memory optimizations
        # "--maxrregcount=64",  # Limit registers per thread
        "--restrict",  # Enable restrict keyword optimization
        "-lineinfo", # Include line info for better debugging
        "-arch=sm_86", # Target Ampere architecture
        # Template-specific optimizations
        # "--diag-suppress=177",  # Suppress unused function warnings from templates
        # "--disable-warnings",  # Disable warnings that might prevent aggressive optimization
        "-t", "0",  # Maximum template instantiation depth
        # "--constant-fold-aggressive",  # Aggressive constant folding for templates
        "luBatchedInplace.cu",
        "-o", "custom"
    ]
    subprocess.run(compile_command, check=True)

def run_ncu_profile(matrix_size: int, num_matrices: int, cublas_dir: str) -> None:
    """Run NCU profiling and save results in the NCU directory."""
    profile_name = f"profile_m{matrix_size}_n{num_matrices}"
    profile_path = os.path.join(cublas_dir, profile_name)
    
    ncu_command = [
        "CUDA_VISIBLE_DEVICES=2",
        "ncu",
        "--set=full",
        "--import-source", "yes",
        "--target-processes", "all",
        "--replay-mode", "kernel",
        "--section", "InstructionStats",
        "--section", "LaunchStats",
        "--section", "MemoryWorkloadAnalysis",
        "--section", "SchedulerStats",
        "--section", "SourceCounters",
        "--section", "SpeedOfLight",
        "--sampling-interval", "auto",
        "--sampling-max-passes", "5",
        "--sampling-buffer-size", "33554432",
        # Additional NCU options
        "--clock-control", "base",  # Lock clock frequency
        "--launch-skip", "0",  # Profile all kernel launches
        "--kernel-name-base", "mangled",  # Use mangled kernel names
        "-f",
        "-o", profile_path,
        "./benchmark"
    ]
    subprocess.run(" ".join(ncu_command), shell=True, check=True)

def run_benchmark(matrix_size: int, num_matrices: int, num_runs: int) -> List[float]:
    """Run benchmark 10 times and collect runtimes."""
    runtimes = []
    incorrect_inversions = 0
    for _ in range(num_runs):
        result = subprocess.run(
            "./custom",
            capture_output=True,
            text=True,
            check=True
        )
        # if the printed lines have "Kernel execution time" in them, then we can extract the runtime from them
        for line in result.stdout.split("\n"):
            if "Kernel execution time" in line:
                runtime_line = (line.split(":")[-1].strip())
                # remove "milliseconds" from the line
                runtime = float(runtime_line.split(" ")[0])
                print(f"Runtime: {runtime}")
                runtimes.append(runtime)
                break

        for line in result.stderr.split("\n"):
            if "Incorrect inversions" in line:
                incorrect_inversions = int(line.split(":")[-1].strip())
                print(f"Incorrect inversions: {incorrect_inversions}")
                break
        # Extract runtime from output (assuming it's printed directly)
        # runtime = float(result.stdout.strip())
        # runtimes.append(runtime)
        time.sleep(1)  # Small delay between runs
    runtime_avg = sum(runtimes) / len(runtimes)
    # calculate standard deviation and variance using numpy
    variance = np.var(runtimes)
    std_dev = np.std(runtimes)
    return runtimes, runtime_avg, variance, std_dev, incorrect_inversions

def save_results(results: dict, filepath: str) -> None:
    """Save benchmark results to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

def main():
    # Parameter ranges
    matrix_sizes = range(1, 33)
    # matrix_sizes = [8, 16, 32]
    # num_matrices_list = [1000, 10000, 100000, 500000, 1000000]
    num_matrices_list = [1000]
    num_runs = 10
    
    # Load required modules
    load_modules()
    
    # Create timestamp and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir, cublas_dir, runtime_dir = create_output_directories(timestamp)
    
    # Store all results
    results = {}
    
    try:
        for matrix_size in matrix_sizes:
            for num_matrices in num_matrices_list:
                    print(f"\nTesting configuration: Matrix Size={matrix_size}, "
                          f"Num Matrices={num_matrices}")
                    
                    # Compile with current parameters
                    compile_cublas(matrix_size, num_matrices)
                    
                    # Run NCU profiling
                    run_ncu_profile(matrix_size, num_matrices, cublas_dir)
                    
                    # Run benchmarks
                    runtimes, runtime_avg, variance, std_dev, incorrect_inversions = run_benchmark(matrix_size, num_matrices, num_runs)
                    
                    # Store results
                    key = f"m{matrix_size}_n{num_matrices}"
                    results[key] = {
                        "matrix_size": matrix_size,
                        "num_matrices": num_matrices,
                        "runtimes": runtimes,
                        "runtime_avg": runtime_avg,
                        "variance": variance,
                        "std_dev": std_dev,
                        "incorrect_inversions": incorrect_inversions
                    }
                    
                    # Save intermediate results
                    runtime_file = os.path.join(runtime_dir, "cublas_results.json")
                    save_results(results, runtime_file)
                    
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        # Save results even if there's an error
        error_file = os.path.join(runtime_dir, "cublas_results_incomplete.json")
        save_results(results, error_file)
    except Exception as e:
        print(f"Unexpected error: {e}")
        error_file = os.path.join(runtime_dir, "cublas_results_incomplete.json")
        save_results(results, error_file)

if __name__ == "__main__":
    main()