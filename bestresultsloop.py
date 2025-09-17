import time
from tqdm import tqdm  # progress bar
import pandas as pd # converting data from txt files 
import torch # creating tensors
from autoemulate.core.compare import AutoEmulate # emulators
import os 

base_path = r"C:\Users\User\Downloads\heart data"
summary_folder = os.path.join(base_path, "summaries")
os.makedirs(summary_folder, exist_ok=True)

model_save_path = "my_emulators"
os.makedirs(model_save_path, exist_ok=True)

start_time = time.time()

for i in tqdm(range(1, 20), desc="Processing datasets"):
    folder = f"{i:02d}"
    x_path = os.path.join(base_path, folder, "X_EP.txt")
    y_path = os.path.join(base_path, folder, "Y.txt")

    # inputs 
    X_df = pd.read_csv(x_path, sep=r"\s+", header=None)
    x = torch.tensor(X_df.values, dtype=torch.float32) # converting to tensor
        
    # output 
    Y_df = pd.read_csv(y_path, sep=r"\s+", header=None)
    y = torch.tensor(Y_df.values, dtype=torch.float32) # converting to tensor
        
    ae = AutoEmulate(x, y, log_level="info") # intialises autoemulate object
    best_em = ae.best_result() # uses built-in function to choose best emulator based on R^2 value 
        
    # prints info about best emulator and run 
    print("Best Emulator :", best_em.model_name) # prints best emulator name
    print("Model ID :", best_em.id) # prints model ID for best run/emulator? 
    print("R^2 (train) : ", best_em.r2_train, "R^2 (test) : ", best_em.r2_test) # prints R^2 
    print("RMSE (train): ", best_em.rmse_train, "RMSE (test) : ", best_em.rmse_test) #prints RMSE

    filename = f"dataset_{folder}_{best_em.model_name}"
    best_result_filepath = ae.save(best_em, model_save_path, use_timestamp=True)
    print("Model and metadata saved to:", best_result_filepath)
      
    best_em_summary = [{ # fills in array with relevant information
        "Dataset": folder,
        "Model Name": best_em.model_name,
        "Model ID": best_em.id,
        "R² Train": round(best_em.r2_train, 4),
        "R² Test": round(best_em.r2_test, 4),
        "RMSE Train": round(best_em.rmse_train, 4),
        "RMSE Test": round(best_em.rmse_test, 4)
        }]
        
    summary_df = pd.DataFrame(best_em_summary) # puts data in a DataFrame (easier to read + easier to save as .csv)
    output_path = os.path.join(summary_folder, f"summary_{folder}.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}") # checks that code has run and .csv file is created

end_time = time.time()
elapsed = end_time - start_time
print(f"\n✅ All datasets processed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
