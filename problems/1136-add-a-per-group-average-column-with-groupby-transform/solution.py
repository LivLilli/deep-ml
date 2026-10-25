import pandas as pd

def solution(df):
    groupby = df.groupby('group').mean().reset_index().rename(columns={'value':'group_avg'})
    output = pd.merge(df, groupby, how='left', on='group')
    return output
