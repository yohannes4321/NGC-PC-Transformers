import os
import sys
import pandas as pd

EXP_DIR = "hebo_experiments"
LOG_DIR = os.path.join(EXP_DIR, "logs")
CSV_PATH = os.path.join(EXP_DIR, "experiments_summary.csv")

os.makedirs(LOG_DIR, exist_ok=True)

def save_to_csv(trial_id, params, metrics):
    row_data = {'trial_id': trial_id}
    # params might be a pandas Series or a dict
    p_dict = params.to_dict() if hasattr(params, 'to_dict') else dict(params)
    row_data.update(p_dict)
    row_data.update(metrics)
    df = pd.DataFrame([row_data])
    write_header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=write_header, index=False)

class DualLogger(object):
    def __init__(self, filepath, terminal_prefix=""):
        self.terminal = sys.stdout
        # buffering=1 ensures line-by-line writing to disk
        self.log = open(filepath, 'w', buffering=1)
        self.prefix = terminal_prefix
        self.at_start_of_line = True

    def write(self, message):
        if message == '\n':
            self.terminal.write(message)
            self.log.write(message)
            self.at_start_of_line = True
        else:
            # If we are at the start of a line, add the worker ID tag [xxxx]
            if self.at_start_of_line and message.strip():
                formatted_msg = f"[{self.prefix}] {message}"
                self.at_start_of_line = False
            else:
                formatted_msg = message
            
            self.terminal.write(formatted_msg)
            self.log.write(message)
        
        # This is the "Heartbeat" fix: force the terminal to show it NOW
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()