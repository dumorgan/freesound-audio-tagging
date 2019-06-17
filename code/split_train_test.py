import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("train_curated_original.csv")

x = train_test_split(df["fname"], df["labels"])
pd.concat([x[0], x[2]], axis=1).to_csv("train_curated.csv", index = False)
pd.concat([x[1], x[3]], axis=1).to_csv("validation_curated.csv", index = False)
