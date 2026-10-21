import pandas as pd

def solution(df1, df2, df3):

    m1 = pd.merge(df1, df2, how='inner', on='emp_id')
    m2 = pd.merge(m1, df3, how='left', on='emp_id')

    return m2


