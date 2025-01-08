import numpy as np
import subprocess
import os
import time
from typing import List, Tuple
import json
from datetime import datetime

def create_output_directories() -> tuple:
    """Create directories for storing NCU profiles and runtime results."""
    if os.path.exists("benchmark_results"):
        subprocess.run(["rm", "-rf", "benchmark_results"])
        
    base_dir = os.path.join("benchmark_results")
    ncu_dir = os.path.join(base_dir, "ncu_profiles")
    runtime_dir = os.path.join(base_dir, "runtime_results")
    
    os.makedirs(ncu_dir, exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)
    
    return base_dir, ncu_dir, runtime_dir

def compile_cuda() -> None:
    """Compile CUDA program with optimization flags."""
    compile_command = [
        "nvcc", "-O3",
        "-g",
        "--resource-usage",
        "--ptxas-options=-v",
        "--expt-relaxed-constexpr",
        "--extra-device-vectorization",
        "--use_fast_math",
        "--default-stream", "per-thread",
        "--std=c++17",
        "--extended-lambda",
        "--expt-extended-lambda",
        "--Werror", "cross-execution-space-call",
        "--dlink-time-opt",
        "--display-error-number",
        "--generate-line-info",
        "--source-in-ptx",
        "-Xcompiler", "-ffast-math",
        "-Xcompiler", "-march=native",
        "-Xcompiler", "-funroll-loops",
        "-Xcompiler", "-fomit-frame-pointer",
        "-Xcompiler", "-ffunction-sections",
        "-Xcompiler", "-fdata-sections",
        "-Xcompiler", "-fno-stack-protector",
        "-Xcompiler", "-fno-math-errno",
        "-Xptxas", "--opt-level=3",
        "-Xptxas", "--allow-expensive-optimizations=true",
        "-Xptxas", "-dlcm=cg",
        "-Xptxas", "-dscm=wt",
        "-Xptxas", "--preserve-relocs",
        "--restrict",
        "-lineinfo",
        "-arch=sm_86",
        "-t", "0",
        "luBatchedInplace.cu",
        "-o", "custom"
    ]
    subprocess.run(compile_command, check=True)
    print("CUDA program compiled successfully")

def run_ncu_profile(matrix_size: int, num_matrices: int, num_threads: int, ncu_dir: str) -> None:
    """Run NCU profiling and save results."""
    profile_name = f"profile_m{matrix_size}_n{num_matrices}_t{num_threads}"
    profile_path = os.path.join(ncu_dir, profile_name)
    
    ncu_command = [
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
        "--clock-control", "base",
        "--launch-skip", "0",
        "--kernel-name-base", "mangled",
        "-f",
        "-o", profile_path,
        "./custom"
    ]
    
    # Create input string for the program
    input_str = f"{matrix_size}\n{num_matrices}\n{num_threads}\n"
    
    # Run NCU with input
    process = subprocess.Popen(ncu_command, stdin=subprocess.PIPE, text=True)
    process.communicate(input_str)
    print(f"NCU profile saved to {profile_path}")

def run_benchmark(matrix_size: int, num_matrices: int, num_threads: int, num_runs: int) -> Tuple[List[float], float, float, float, List[int]]:
    """Run benchmark multiple times and collect metrics."""
    runtimes = []
    incorrect_inversions = []
    
    for run in range(num_runs):
        print(f"Run number: {run + 1}")
        
        # Create input string
        input_str = f"{matrix_size}\n{num_matrices}\n{num_threads}\n"
        
        # Run the program
        process = subprocess.Popen(
            "./custom",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input_str)
        
        # Extract runtime
        for line in stdout.split("\n"):
            if "Kernel execution time" in line:
                runtime = float(line.split(":")[-1].strip().split(" ")[0])
                runtimes.append(runtime)
                break
        
        # Extract incorrect inversions
        for line in stderr.split("\n"):
            if "Incorrect inversions" in line:
                temp = int(line.split(":")[-1].strip().split(" ")[-1].strip())
                if temp > 0:
                    incorrect_inversions.append(temp)
                break
        
        time.sleep(1)  # Small delay between runs
    
    runtime_avg = np.mean(runtimes)
    variance = np.var(runtimes)
    std_dev = np.std(runtimes)
    return runtimes, runtime_avg, variance, std_dev, incorrect_inversions

def save_results(results: dict, filepath: str) -> None:
    """Save benchmark results to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

def get_num_threads(matrix_size: int) -> int:
    """Calculate optimal number of threads based on matrix size."""
    if matrix_size <= 16:
        return 32
    elif matrix_size in [3, 5, 6, 10, 15]:
        return 30
    elif matrix_size in [7, 14]:
        return 28
    elif matrix_size == 9:
        return 27
    elif matrix_size == 11:
        return 22
    elif matrix_size == 12:
        return 24
    elif matrix_size == 13:
        return 26
    else:
        return matrix_size

def main():
    # Parameter ranges
    matrix_sizes = range(1, 33)
    num_matrices_list = [100000, 500000, 1000000]
    # num_matrices_list = [100, 500, 1000]
    matrix_names = ["100k", "500k", "1M"]
    num_runs = 10
    
    # Create directories
    base_dir, ncu_dir, runtime_dir = create_output_directories()
    
    try:
        # Compile CUDA program once at the start
        compile_cuda()
        
        for num_matrices, matrix_name in zip(num_matrices_list, matrix_names):
            file_name = f"benchmark_results_{matrix_name}.json"
            ncu_matrix_dir = os.path.join(ncu_dir, matrix_name)
            runtime_matrix_dir = os.path.join(runtime_dir, matrix_name)
            os.makedirs(ncu_matrix_dir, exist_ok=True)
            os.makedirs(runtime_matrix_dir, exist_ok=True)
            results = {}

            for matrix_size in matrix_sizes:
                print(f"\nTesting configuration: Matrix Size={matrix_size}, Num Matrices={num_matrices}")
                
                # Calculate number of threads
                num_threads = get_num_threads(matrix_size)
                print(f"Number of threads: {num_threads}")
                
                # Run NCU profiling
                run_ncu_profile(matrix_size, num_matrices, num_threads, ncu_matrix_dir)
                
                # Run benchmarks
                runtimes, runtime_avg, variance, std_dev, incorrect_inversions = run_benchmark(
                    matrix_size, num_matrices, num_threads, num_runs
                )
                
                # Store results
                key = f"m{matrix_size}_n{num_matrices}_t{num_threads}"
                results[key] = {
                    "matrix_size": matrix_size,
                    "num_matrices": num_matrices,
                    "num_threads": num_threads,
                    "runtimes": runtimes,
                    "runtime_avg": runtime_avg,
                    "variance": variance,
                    "std_dev": std_dev,
                    "incorrect_inversions": incorrect_inversions
                }
                
                # Save intermediate results
                runtime_file = os.path.join(runtime_matrix_dir, file_name)
                save_results(results, runtime_file)
                
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        error_file = os.path.join(runtime_dir, "benchmark_results_incomplete.json")
        save_results(results, error_file)
    except Exception as e:
        print(f"Unexpected error: {e}")
        error_file = os.path.join(runtime_dir, "benchmark_results_incomplete.json")
        save_results(results, error_file)

if __name__ == "__main__":
    main()