import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv(r"C:\Users\User\Downloads\research project\heart data\allmodelsummaries\averages.csv")
print(df.columns)

plt.figure(figsize=(10,10))

plt.errorbar(
    x = range(len(df)), 
    y = df["r2 "], 
    yerr = df["r2_std"], 
    fmt = 'o',                
    capsize = 5,             
    elinewidth = 1.5, 
    markersize = 4, 
    color = "black", 
    ecolor = "black"
)

plt.xlabel("Model")
plt.ylabel("R²")
plt.title("R² values with error bars")
plt.tight_layout()

plt.xticks(range(len(df)), df.iloc[:, 0])

plt.savefig(r"C:\Users\User\Downloads\research project\plots\r2_plot.png", dpi=300)
plt.show()




