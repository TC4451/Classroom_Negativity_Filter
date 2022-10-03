import pandas as pd
import numpy as np
import glob
import os

# concatenate all result csv from time_interval_analysis to a single csv, all of which text is detected
files = glob.glob('../results/prev_results/*', recursive=True)
df = pd.concat([pd.read_csv(fp).assign(file_name=os.path.basename(fp).split('.')[0]) for fp in files])
df = df.iloc[: , 1:]
# make sure all rows have text in it
df['score'].replace('', np.nan, inplace=True)
df.dropna(subset=['score'], inplace=True)
df.to_csv("../concatenated_csv.csv")
