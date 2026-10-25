import pandas as pd


def solution(df):
    # TODO: bucket scores, count per (grade, bucket), cumulative percentage within grade
    def compute_bucket(x):
        return (x // 10) * 10

    # TODO: bucket scores, count per (grade, bucket), cumulative percentage within grade
    df['bucket'] = df['score'].apply(compute_bucket)
    result = df.groupby(['grade', 'bucket']).count().reset_index().rename(columns={'score':'count'})
    result = result.sort_values(['grade', 'bucket'])
    result['cum_pct'] = (result.groupby('grade')['count'].cumsum()/ result.groupby('grade')['count'].transform('sum')* 100).round(2)    

    return result