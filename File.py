#%%
import pandas as pd

pd.set_option('display.max_columns', 0)
df = pd.read_csv('data/train.csv', index_col='Id')
df.head(5)
