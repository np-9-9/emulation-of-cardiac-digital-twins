import pandas as pd 
import os
import matplotlib.pyplot as plt 

base_path = base_path = r"C:\Users\User\Downloads\research project\heart data 3"

data2plot = os.path.join(base_path, r"case01\overallsummary.csv")
data_df = pd.read_csv(data2plot)

print(data_df.columns)

x_pos = range(len(data_df))

plt.errorbar(
    x = x_pos, 
    y = data_df["r2_test"], 
    yerr = data_df["r2_test_std"], 
    color = "m", 
    fmt = "o", 
    markersize = 8,
    capsize = 3
)

output_loc = os.path.join(base_path, r"case01\modelplots.png")
plt.title("r^2 for different models for case 1")
plt.xlabel("Models")
plt.ylabel("r^2 value")
plt.xticks(x_pos, data_df.iloc[:, 0])
plt.savefig(output_loc, dpi = 300)
plt.show() 



