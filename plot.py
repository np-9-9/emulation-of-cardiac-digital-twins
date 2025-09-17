import pandas as pd 
import os
import matplotlib.pyplot as plt 

base_path = r"C:\Users\User\Downloads\research project\heart data 3"

polychaos_data = os.path.join(base_path, "polychaosr2.csv")
pcdata = pd.read_csv(polychaos_data, sep=r"\s+", header=None).values
x = list(range(1, 11))
y = pcdata[:, 0]

plt.scatter(x, y, color = "m")
plt.xlabel("Case Number")
plt.xticks(x)
plt.ylabel("R^2 values")
plt.savefig(r"C:\Users\User\Downloads\research project\heart data 3\scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()