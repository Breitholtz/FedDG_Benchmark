import json
import subprocess
import copy
from options import args_parser
import sys

def run_experiment(config_file):
    with open(config_file) as fh:
        config = json.load(fh)
    
    dirichlet_beta_values = [0.5, 1.0, 5.0, 10.0]
    for dirichlet_beta_value in dirichlet_beta_values:
        config["dirichlet_beta"] = dirichlet_beta_value
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