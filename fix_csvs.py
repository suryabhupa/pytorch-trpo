import numpy as np
import os

filenames = [
"logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-10.csv",
"logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-20.csv",
"logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-30.csv",
"logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-40.csv", 
"logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-50.csv",
"logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-10.csv",
"logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-20.csv",
"logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-30.csv",
"logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-40.csv",
"logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-50.csv",
"logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-10.csv",
"logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-20.csv",
"logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-30.csv",
"logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-40.csv",
"logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-50.csv",
]

for filename in filenames[10:15]: 
  with open(filename, 'r+') as f:
    lines = f.read().split('\n')
    f.seek(0)
    f.truncate()
    f.write(lines[0] + "\n")
    f.write(lines[0] + "\n")
    entries = lines[1].split(',')
    for i in range(0, len(entries), 3):
      f.write(",".join(entries[i:i+3]) + "\n")

