import pandas as pd

def solution(df):
    df['name'] = df['name'].str.strip().str.title() #step 1
    df['date'] = pd.to_datetime(df['date'].str.strip()).dt.strftime('%Y-%m-%d') # step 2
    df = df.drop_duplicates(keep='first').reset_index(drop=True) # step 3
    df['value'] = df['value'].fillna(df['value'].mean()) # step 4


    return df
