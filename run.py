import os
import subprocess
from glob import glob
import time

def run_all_scripts():
    # Get all directories in current path
    directories = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    # Store execution results
    results = []
    
    for directory in directories:
        run_file = os.path.join(directory, 'run.py')
        
        # Check if run.py exists in the directory
        if os.path.exists(run_file):
            print(f"\nStarting {directory}/run.py...")
            
            # Create log file name
            log_file = f"nohup_{directory}.out"
            
            # Open log file
            with open(log_file, 'w') as f:
                # Run the script with nohup and wait for it to complete
                process = subprocess.Popen(
                    ['python3', 'run.py'],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=directory
                )
                
                # Record start time
                start_time = time.time()
                
                # Wait for the process to complete
                process.wait()
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Store result
                results.append({
                    'directory': directory,
                    'exit_code': process.returncode,
                    'log_file': log_file,
                    'execution_time': execution_time
                })
                
                print(f"Finished {directory}/run.py")
                print(f"Execution time: {execution_time:.2f} seconds")
                print(f"Exit code: {process.returncode}")
    
    # Print summary
    print("\nExecution Summary:")
    print("-" * 50)
    for r in results:
        print(f"Directory: {r['directory']}")
        print(f"Exit Code: {r['exit_code']}")
        print(f"Execution Time: {r['execution_time']:.2f} seconds")
        print(f"Log File: {r['log_file']}")
        print("-" * 50)

if __name__ == "__main__":
    run_all_scripts()