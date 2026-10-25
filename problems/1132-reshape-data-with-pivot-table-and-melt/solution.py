import pandas as pd

def solution(df):
    prod_values = list(set(list(df['product'])))

    pivot = pd.pivot_table(
        df,
        index='date',
        columns = 'product',
        values='sales',
        aggfunc='sum',
        fill_value=0

    ).reset_index()#.rename(columns={'product':'date'})
    return pd.melt(pivot, id_vars = 'date', value_vars = prod_values).sort_values(['date', 'product']).rename(columns={'value':'sales'})[['date', 'product', 'sales']]
