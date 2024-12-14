from scipy.stats import ttest_ind
import pandas as pd

df = pd.read_csv("results.csv")

hv_col = df["horizontal-vertical"]
diag_col = df["diagonal"]

t_stat, p_value = ttest_ind(hv_col, diag_col, equal_var=False)

print(t_stat)
print(p_value)
