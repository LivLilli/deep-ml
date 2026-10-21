import pandas as pd

def solution(df):
    
    df_filtered = df[df.status == 'completed']
    df_out = df_filtered[['region', 'amount']].groupby('region').sum().reset_index()

    return df_out.sort_values(['amount', 'region'], ascending=[ False, True]).head(10)
