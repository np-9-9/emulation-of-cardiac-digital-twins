import time
from tqdm import tqdm  # progress bar
import pandas as pd # converting data from txt files 
import torch # creating tensors
from autoemulate.core.compare import AutoEmulate # emulators
import os

start_time = time.time()

base_path = r"C:\Users\User\Downloads\research project\heart data"

for i in tqdm(range(1, 20), desc="Processing datasets"):
    folder_name = f"{i:02d}"
    folder_path = os.path.join(base_path, folder_name)

    x_path = os.path.join(folder_path, "X_EP.txt")
    y_path = os.path.join(folder_path, "Y.txt")
    summary_path = os.path.join(folder_path, "summary.csv")

    # inputs 
    X_df = pd.read_csv(x_path, sep=r"\s+", header=None)
    x = torch.tensor(X_df.values, dtype=torch.float32) # converting to tensor

    # output 
    Y_df = pd.read_csv(y_path, sep=r"\s+", header=None)
    y = torch.tensor(Y_df.values, dtype=torch.float32) # converting to tensor

    # creating autoemulate object 
    ae = AutoEmulate(x, y, log_level="info")

    print("summarise starting")
    summary = ae.summarise()
    print(summary)

    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

end_time = time.time()
print(f"\nAll summaries completed in {end_time - start_time:.2f} seconds.")




