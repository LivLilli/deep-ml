import pandas as pd

def solution(df):
    
    df = df.loc[:, df.isna().mean() <= 0.5]
    df = df.loc[df.isna().mean(axis=1) <= 0.5]   

    num_cols = list(df.select_dtypes(include='number').columns)
    cat_cols = list(df.select_dtypes(include=['string', 'object']).columns)
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df