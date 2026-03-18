import pandas as pd

# Load your dataset
df = pd.read_csv("dataset.csv")

# Check distribution of very_high_reach
print(df["very_high_reach"].value_counts(normalize=True))

print("Counts:")
print(df["very_high_reach"].value_counts())

print("\nPercentages:")
print(df["very_high_reach"].value_counts(normalize=True))