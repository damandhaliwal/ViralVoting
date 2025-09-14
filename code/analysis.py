from utils import get_project_paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = get_project_paths()

data = pd.read_csv(path['data'] + '/social.csv')

# print number of rows and columns
print (data.head)
