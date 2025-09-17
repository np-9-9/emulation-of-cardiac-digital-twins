import pandas as pd # converting data from txt files 
import torch # creating tensors
from autoemulate.core.compare import AutoEmulate # emulators
import matplotlib.pyplot as plt # plotting graphs

# inputs 
x_path = r"C:\Users\User\Downloads\heart data\01\X_EP.txt"
X_df = pd.read_csv(x_path, sep=r"\s+", header=None)
x = torch.tensor(X_df.values, dtype=torch.float32) # converting to tensor

# output 
y_path = r"C:\Users\User\Downloads\heart data\01\Y.txt"
Y_df = pd.read_csv(y_path, sep=r"\s+", header=None)
y = torch.tensor(Y_df.values, dtype=torch.float32) # converting to tensor

print(x.shape, y.shape) # checking tensor dimensions

ae = AutoEmulate(x, y, log_level="error") # intialises autoemulate object 

# prints best result for each emulator (AutoEmulate runs emulators multiple times changing some things?)
summary = [] # initialises empty array 
for r in ae.results: # loops over all emulator runs 
    if r.model_name not in [s[0] for s in summary]: # avoids duplicates 
        emulator_results = [res for res in ae.results if res.model_name == r.model_name] # looks at all results for ONE emulator 
        best_run = min(emulator_results, key=lambda res: res.rmse_test) # takes run with minimum rmse (best)
        summary.append((best_run.model_name, best_run.r2_test, best_run.rmse_test, best_run.model)) # adds best run to summary 

print("\nFirst 5 predictions from the best run of each emulator:")
for name, r2, rmse, model in summary:
    y_pred_em = model.predict(x)

    # handle distribution outputs
    if hasattr(y_pred_em, "mean"):
        y_pred_em = y_pred_em.mean()

    if isinstance(y_pred_em, torch.Tensor):
        y_pred_em = y_pred_em.detach().numpy()

    print(f"{name:25} R²={r2:.3f}, RMSE={rmse:.3f}")
    print(y_pred_em[:5])   # first 5 predictions
    print("-" * 60)

# best model (comes from AutoEmulate class - chooses best emulator run over all emulator types)
best_result = ae.best_result()
best_model = best_result.model
y_pred = best_model.predict(x) 

# takes mean to be prediction if emulator method gives a distribution 
if hasattr(y_pred, "mean"):
    y_pred = y_pred.mean()

# converts pytorch tensors to numpy arrays 
if isinstance(y_pred, torch.Tensor):
    y_pred = y_pred.detach().numpy()

# prints first few predictions from best model overall 
print("\nFirst 5 predictions from the best model:")
print(y_pred[:5])

# --- Plot predicted vs true for each output dimension ---
y_true = y.detach().numpy() if isinstance(y, torch.Tensor) else y
n_outputs = y_true.shape[1]

for i in range(n_outputs):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true[:, i], y_pred[:, i])
    plt.plot([y_true[:, i].min(), y_true[:, i].max()],
             [y_true[:, i].min(), y_true[:, i].max()],
             'r--', lw=2)  # identity line
    plt.xlabel("True Output")
    plt.ylabel("Predicted Output")
    plt.title(f"Output {i+1}: Predicted vs True")
    plt.grid(True)
    plt.show()

