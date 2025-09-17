import pandas as pd # converting data from txt files 
import torch # creating tensors
from autoemulate.core.compare import AutoEmulate # emulators
import os 

base_path = r"C:\Users\User\Downloads\research project\heart data 3"

for i in range(1, 11) : 
      # inputs
      exact_x = os.path.join(base_path, f"case{i:02}", "X.txt")
      X_df = pd.read_csv(exact_x, sep=r"\s+", header=None)
      x = torch.tensor(X_df.values, dtype=torch.float32) # converting to tensor
      
      # output 
      y_path = os.path.join(base_path, f"case{i:02}", "split_outputs")
      
      for j in range(1, 8) : 
            exact_y_path = os.path.join(y_path, f"Y_col{j}.txt")
            Y_df = pd.read_csv(exact_y_path, sep=r"\s+", header=None)
            y = torch.tensor(Y_df.values, dtype=torch.float32) # converting to tensor
            
            ae = AutoEmulate(x, y, log_level="info") # intialises autoemulate object 
            
            summary_path = os.path.join(y_path, f"summary_case{i:02}_col{j}.csv")
            print("summarise starting")
            summary = ae.summarise()
            print(summary)
            
            summary.to_csv(summary_path, index=False)
            print(f"Saved summary to: {summary_path}")
