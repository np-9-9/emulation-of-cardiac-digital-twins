import pandas as pd # converting data from txt files 
import torch # creating tensors
from autoemulate.core.compare import AutoEmulate # emulators
import os 

# inputs 
x_path = r"C:\Users\User\Downloads\research project\heart data 3\case01\X.txt"
x_df = pd.read_csv(x_path, sep=r"\s+", header=None)
x = torch.tensor(x_df.values, dtype = torch.float32) # conevrting to tensor

# outputs 
y_path =  r"C:\Users\User\Downloads\research project\heart data 3\case01\Y.txt"
y_df = pd.read_csv(y_path, sep=r"\s+", header=None)
y = torch.tensor(y_df.values, dtype = torch.float32) # conevrting to tensor

ae = AutoEmulate(x, y, log_level="info")

summary_path = r"C:\Users\User\Downloads\research project\heart data 3\case01\overallsummary.csv"

print("summary starting")
summ = ae.summarise()

summ.to_csv(summary_path, index=False)
print(f"Saved summary to: {summary_path}")
