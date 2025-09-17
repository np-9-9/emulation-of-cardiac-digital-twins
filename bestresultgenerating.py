import pandas as pd # converting data from txt files 
import torch # creating tensors
from autoemulate.core.compare import AutoEmulate # emulators

# inputs 
x_path = r"C:\Users\User\Downloads\heart data\01\X_EP.txt"
X_df = pd.read_csv(x_path, sep=r"\s+", header=None)
x = torch.tensor(X_df.values, dtype=torch.float32) # converting to tensor

# output 
y_path = r"C:\Users\User\Downloads\heart data\01\Y.txt"
Y_df = pd.read_csv(y_path, sep=r"\s+", header=None)
y = torch.tensor(Y_df.values, dtype=torch.float32) # converting to tensor

ae = AutoEmulate(x, y, log_level="info") # intialises autoemulate object 
best_em = ae.best_result() #uses built-in function to choose best emulator based on R^2 value 

# prints info about best emulator and run 
print("Best Emulator :", best_em.model_name) # prints best emulator name
print("Model ID :", best_em.id) # prints model ID for best run/emulator? 
print("R^2 (train) : ", best_em.r2_train, "R^2 (test) : ", best_em.r2_test) # prints R^2 
print("RMSE (train): ", best_em.rmse_train, "RMSE (test) : ", best_em.rmse_test) #prints RMSE