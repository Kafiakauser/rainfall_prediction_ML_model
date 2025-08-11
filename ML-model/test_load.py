import pandas as pd

file_path = r"C:\Users\Hp\OneDrive\Documents\ML-model\weatherAUS.csv"

# Load CSV
df = pd.read_csv(file_path)

# Quick check
print(df.shape)      # rows, columns
print(df.head())     # first 5 rows
