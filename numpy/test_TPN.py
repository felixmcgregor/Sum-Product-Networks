import numpy as np
import pandas as pd
from learnSPN import learnSPN


# generate new dataset
data = []

# paper dataset
data.append([0.0, 0.0, 1.0, 0.0])
data.append([0.0, 0.0, 0.0, 1.0])
data.append([0.0, 1.0, 0.0, 0.0])
data.append([0.0, 1.0, 0.0, 1.0])
data.append([0.0, 1.0, 1.0, 0.0])
data.append([1.0, 0.0, 0.0, 0.0])
data.append([1.0, 0.0, 0.0, 1.0])
data.append([1.0, 0.0, 1.0, 0.0])
data.append([1.0, 0.0, 1.0, 1.0])
data.append([1.0, 1.0, 1.0, 0.0])

# XOR dataset
data = []
x1 = np.array([0.0, 0.0, 0.0])
x2 = np.array([0.0, 1.0, 0.0])
x3 = np.array([1.0, 0.0, 0.0])
x4 = np.array([1.0, 1.0, 0.0])

x5 = np.array([0.0, 0.0, 1.0])
x6 = np.array([0.0, 1.0, 1.0])
x7 = np.array([1.0, 0.0, 1.0])
x8 = np.array([1.0, 1.0, 1.0])
data = np.array([x1, x2, x3, x4, x5, x6, x7, x8])

data = np.array(data)


df = pd.DataFrame(data)
new_names = []
old_names = df.columns
for i in old_names:
    df.rename(columns={i: ('X' + str(i))}, inplace=True)

learnt_spn = learnSPN(df)
learnt_spn.normalise_weights()
learnt_spn.print_weights()
