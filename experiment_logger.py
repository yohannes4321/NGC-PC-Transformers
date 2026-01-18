import os
import sys
import pandas as pd

EXP_DIR = "hebo_experiments"
LOG_DIR = os.path.join(EXP_DIR, "logs")
CSV_PATH = os.path.join(EXP_DIR, "experiments_summary.csv")

os.makedirs(LOG_DIR, exist_ok=True)

def save_to_csv(trial_id, params, metrics):
    row_data = {'trial_id': trial_id}
    row_data.update(params.to_dict())
    row_data.update(metrics)
    df = pd.DataFrame([row_data])
    write_header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=write_header, index=False)

class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', buffering=1) # Line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()