import pandas as pd
import math
import numpy as np
from typing import List

def td(amount, unit='s'):
    return pd.Timedelta(f"{amount}{unit}")

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

def split_array_into_chunks(array: np.ndarray, n_chunks: int) -> List[np.ndarray]:
    """
    Split a numpy array into N evenly spaced chunks.
    
    Args:
        array (np.ndarray): The input array to split
        n_chunks (int): Number of chunks to split the array into
        
    Returns:
        List[np.ndarray]: List containing the chunked arrays
        
    Example:
        >>> arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> chunks = split_array_into_chunks(arr, 3)
        >>> [len(chunk) for chunk in chunks]
        [4, 3, 3]
    """
    total_length = len(array)
    chunk_size = total_length // n_chunks
    remainder = total_length % n_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(n_chunks):
        # Calculate the end index for this chunk
        extra = 1 if i < remainder else 0
        end_idx = start_idx + chunk_size + extra
        
        # Add the chunk to our list
        chunks.append(array[start_idx:end_idx])
        
        # Update start index for next chunk
        start_idx = end_idx
    
    return chunks