import pandas as pd
import numpy as np

df = pd.read_csv("weatherHistory.csv")

#print all the cols
print(df.columns)

# keep only the col Temperature (C) and save as a csv file
df = df[['Temperature (C)']]
df.to_csv('temperature.csv', index=False)


