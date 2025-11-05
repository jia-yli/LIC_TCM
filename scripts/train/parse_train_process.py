import os
import re
import pandas as pd

def main(input_path, min_job_id):
  '''
  Find Files
  '''
  all_files = sorted([f for f in os.listdir(input_path) if f.endswith('.txt')])

  file_pattern = re.compile(r'job_(\d+)_out.txt')
  filtered_files = [
    f for f in all_files
    if (match := file_pattern.match(f)) and (int(match.group(1)) >= min_job_id)
  ]

  '''
  Gather Data
  '''
  data = []
  variable_pattern = re.compile(r'^variable\s:\s(\S*)$')
  save_path_pattern = re.compile(r'^save_path\s:\s(\S*)$')
  valid_pattern = re.compile(r'Valid epoch (\d+): Average losses:\s*Loss:\s*([\d.]+)\s*\|\s*MSE loss:\s*([\d.]+)\s*\|\s*Bpp loss:\s*([\d.]+)\s*\|\s*Aux loss:\s*([\d.]+)')
  
  for fname in filtered_files:
    with open(os.path.join(input_path, fname), 'r') as f:
      variable = None
      save_path = None
      for line in f:
        if variable_match := variable_pattern.match(line):
          variable = variable_match.group(1)
        elif save_path_match := save_path_pattern.match(line):
          save_path = save_path_match.group(1)
        elif valid_match := valid_pattern.match(line):
          data.append({
            "variable" : variable,
            "save_path" : save_path,
            "epoch" : valid_match.group(1),
            "loss" : valid_match.group(2),
            "mse_loss" : valid_match.group(3),
            "bpp_loss" : valid_match.group(4),
            "aus_loss" : valid_match.group(5),
          })

  df = pd.DataFrame(data)
  df.to_csv("./results/tcm_weighted/train_process_freeze.csv", index=False)
  import pdb;pdb.set_trace()

  '''
  Plot
  '''

if __name__ == "__main__":
  input_path = "/users/ljiayong/projects/LIC_TCM/slurm_out"
  min_job_id = 606433
  main(input_path, min_job_id)