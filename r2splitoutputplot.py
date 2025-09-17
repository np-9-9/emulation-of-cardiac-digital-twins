import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

long_to_short = {"GaussianProcess" : "GP", 
                 "GaussianProcessCorrelated" : "GPCorr", 
                 "EnsembleMLP" : "E-MLP",
                 "MLP" : "MLP",
                 "EnsembleMLPDropout" : "E-MLP-D", 
                 "RadialBasisFunctions" : "RBF", 
                 "RandomForest" : "RF", 
                 "SupportVectorMachine" : "SVM", 
                 "LightGBM" : "L-GBM"
}

for i in range(1, 8) : 
    base_path = r"C:\Users\User\Downloads\research project\heart data 3"
    split_path = os.path.join(base_path, f"summary0{i}.csv")
    split_path_df = pd.read_csv(split_path)

    x_pos = range(len(split_path_df))

    plt.errorbar(
        x = x_pos,
        y = split_path_df["r2_test"],
        yerr = split_path_df["r2_test_std"],
        fmt = "o",
        color = "m", 
        markersize = 7,
        capsize = 5
    )

    labels = split_path_df["model_name"].map(long_to_short).fillna(split_path_df["model_name"]).tolist()

    plt.xticks(range(len(split_path_df)), labels)
    plt.xlabel("Model")
    plt.ylabel("r2")
    plt.title("r2 over all models")
    plt.tight_layout()

    save_path = os.path.join(base_path, f"r2plot{i}.png")

    plt.savefig(save_path, dpi=300)
    plt.close()