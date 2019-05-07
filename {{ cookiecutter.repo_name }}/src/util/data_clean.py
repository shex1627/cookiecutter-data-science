####
# data type check
####
import pandas as pd
import numpy as np

def check_na(df, norm=False):
    """quick check the missing value stats for each column in a df."""
    if norm:
        return df.isna().sum()/df.shape[0]
    else:
        return df.isna().sum()

def make_check_type(dtype):
    """check if an element can be converted the desire dtype"""

    def check_type(element):
        try:
            dtype(element)
            return True
        except ValueError:
            return False

    return check_type


def dtype_check(df, df_dtype):
    """
    check if elements in the dataframe fits the type described by df_type.

    Args:
        df (pandas dataframe): dataframe to check type from
        df_dtype (dict) : a dictionary maps dataframe columns to its corresponding supposed data types

    Returns:
        dtype_check: pandas dataframe with same shape as df, with boolean if each data fits the type requirement
        dtype_check_stat: pandas series, index is df's column names, value is number of rows don't fit the required data
        type.
        rows_to_keep: pandas series, true or false, indicating if row from df fits all datatype requirement

    Examples:
        >>> dtype_check_stat, rows_to_keep = dtype_check(data_frame, file_dtype)[1:]

    """

    dtype_check = {}
    for col, col_type in df_dtype.items():
        dtype_check[col] = df[col].apply(make_check_type(col_type))

    dtype_check = pd.DataFrame(dtype_check)
    dtype_check_stat = df.shape[0] - dtype_check.apply(np.sum, axis=0)
    rows_to_keep = dtype_check.apply(all, axis=1)

    return (dtype_check, dtype_check_stat, rows_to_keep)

