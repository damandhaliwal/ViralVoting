import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('social.csv')

# find out how many rows in data
num_rows = data.shape[0]
print(f"Number of rows in the dataset: {num_rows}")