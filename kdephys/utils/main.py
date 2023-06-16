import pandas as pd
import math


def prev_day(t):
    return t - pd.Timedelta("1D")


def time_zone_df(df, dur=1800):
    """Chunk a dataframe into time chunks of dur seconds.
    Note that df must have a datetime column.
    """
    assert type(df) == pd.core.frame.DataFrame, "df must be a pandas dataframe"
    assert "datetime" in df.columns, "df must have a datetime column"
    diff = (df.datetime.max() - df.datetime.min()).total_seconds()
    n = math.ceil(diff / dur)
    for i in range(n):
        start = df.datetime.min() + pd.Timedelta(f"{i*dur}s")
        end = start + pd.Timedelta(f"{dur}s")
        df.loc[((df.datetime >= start) & (df.datetime < end)), "time_chunk"] = i + 1
    return df
