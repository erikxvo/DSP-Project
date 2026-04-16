
import pandas as pd

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Group by sex and calculate survival rate
survival_rate = df.groupby("Sex")["Survived"].mean()

# Print results
print(survival_rate)

