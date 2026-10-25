import pandas as pd
import numpy as np


def solution(df, window):
    # TODO: complete the daily calendar, keep NaN for missing days, trailing calendar-window mean
    df = df.sort_values('date').set_index('date')

    # Create a complete range of calendar days

    all_dates = pd.date_range(

        start=df.index.min(),

        end=df.index.max(),

        freq='D'

    )

    # Add missing days (value = NaN)

    df = df.reindex(all_dates)

    # Compute rolling mean

    df['rolling_mean'] = df['value'].rolling(

        window=window,

        min_periods=1

    ).mean()

    return df.rename_axis('date').reset_index()