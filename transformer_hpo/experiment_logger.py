# filename: experiment_logger.py
import os
import sys
import pandas as pd

# --- CONFIGURATION ---
EXP_DIR = "hebo_experiments"
LOG_DIR = os.path.join(EXP_DIR, "logs")
CSV_PATH = os.path.join(EXP_DIR, "experiments_summary.csv")

# Create directories immediately upon import
os.makedirs(LOG_DIR, exist_ok=True)

def save_to_csv(trial_id, params, metrics):
    """
    Appends a single row to the master CSV file.
    """
    # Combine params and metrics into one dictionary
    row_data = {'trial_id': trial_id}
    row_data.update(params.to_dict()) # Hyperparams
    row_data.update(metrics)          # Results (CL, PPL, EFE)
    
    df = pd.DataFrame([row_data])
    
    # Append to CSV (header only for the first write)
    write_header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=write_header, index=False)

class DualLogger(object):
    """
    A helper class to print to both the Console and a Text File simultaneously.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()