import pandas as pd
import json

# 생략없이 다 보이게
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("../atomic/v4_atomic_all.csv")
test_df = df[df['event'] == 'PersonX repels PersonY attack']
print(test)