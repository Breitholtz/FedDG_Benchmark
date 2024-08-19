import json
import subprocess
import copy
from options import args_parser
import sys

def run_experiment(config_file):
    with open(config_file) as fh:
        config = json.load(fh)
    
    lr_list = [0.1, 0.01, 0.001]
    #lr_alpha_list = [0.1, 0.01, 0.001]
    lr_alpha_list = [0.1]
    n_experiments = len(lr_list) * len(lr_alpha_list)
    count = 0
    for lr in lr_list:
        for lr_alpha in lr_alpha_list:
            config["lr_alpha"] = lr_alpha
            config["lr"] = lr
            config_file = "./config_hparam_temp.json"
            count += 1
            print(f"Hparam: lr: {lr} | lr_alpha: {lr_alpha}, experiment {count}/{n_experiments}")
            # Save the modified configuration to a temporary file
            with open(config_file, 'w') as fh:
                json.dump(config, fh)
            
            # Run the experiment with the modified configuration
            #print("run main")
            result = subprocess.run(["python", "hparam_search.py", "--config_file", config_file], capture_output=True, text=True)
            #print(result.stdout)
            print(result.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_hparam_search.py <config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    
    print(f"Using configuration file: {config_file_path}")
    
    # Run the experiments
    run_experiment(config_file_path)