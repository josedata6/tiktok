import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# load data
df = pd.read_csv("dataset.csv")

# quick peek
# print(df.head())
# print(df.info())

######## summary statistics ########
# basic stats
summary = df.describe(include="all")
print(summary)

# missing values
missing = df.isnull().mean().sort_values(ascending=False)
print(missing)

######## END summary statistics ########

#### distribution plots #####
numeric_cols = [
    "follower_count",
    "authority_log",
    "early_total_engagement",
    "reach",
    "impressions",
    "total_likes",
    "total_comments",
    "total_shares",
    "link_clicks",
    "saves"
]

for col in numeric_cols:
    plt.figure()
    df[col].dropna().plot(kind="hist", bins=50)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
#### END distribution plots #####

### Early Signals vs Final Outcomes###

plt.figure()
plt.scatter(
    df["early_total_engagement"],
    df["total_likes"] + df["total_comments"] + df["total_shares"],
    alpha=0.4
)
plt.xlabel("Early Total Engagement")
plt.ylabel("Final Total Engagement")
plt.title("Early vs Final Engagement")
plt.show()

#### END Early Signals vs Final Outcomes ###

### correlation heatmap ###

corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr)
plt.colorbar()
plt.title("Feature Correlation Matrix")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
#### END correlation heatmap ###