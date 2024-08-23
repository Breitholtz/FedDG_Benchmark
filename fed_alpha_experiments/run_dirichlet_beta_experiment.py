import json
import subprocess
import copy
from options import args_parser
import sys

def run_experiment(config_file):
    with open(config_file) as fh:
        config = json.load(fh)
    
    #dirichlet_beta_values = [0.5, 1.0, 3.0, 5.0, 10]
    #lr_values = [1e-4, 5e-4, 1e-3, 1e-2, 1e-1]
    reg_lambda_values = [0] #[10, 100, 1000, 10000]
    for value in reg_lambda_values:
        config["reg_lambda"] = value
        config_file = "./config_temp.json"
        
        # Save the modified configuration to a temporary file
        with open(config_file, 'w') as fh:
            json.dump(config, fh)
        
        # Run the experiment with the modified configuration
        print("run main")
        result = subprocess.run(["python", "main.py", "--config_file", config_file], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_dirichlet_beta_experiment.py <config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    
    print(f"Using configuration file: {config_file_path}")
    
    # Run the experiments
    run_experiment(config_file_path)