import pandas as pd
import os

# Input file
input_base = r"C:\Users\User\Downloads\research project\heart data 3"

for i in range(1, 11) : 
    exact_basefile = os.path.join(input_base, f"case{i:02}", "Y.txt")
    df = pd.read_csv(exact_basefile, sep=r"\s+", header=None) # read file (space separated)

    output_base = r"C:\Users\User\Downloads\research project\heart data 3"
    exact_output = os.path.join(output_base, f"case{i:02}", "split_outputs")
    os.makedirs(exact_output, exist_ok=True)
    
    # Save each column to its own file
    for j, col in enumerate(df.columns):
        output_path = os.path.join(exact_output, f"Y_col{j+1}.txt")
        df[[col]].to_csv(output_path, sep="\t", header=False, index=False)
        print(f"Saved {output_path}")