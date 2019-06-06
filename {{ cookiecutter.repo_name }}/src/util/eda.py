import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cdfpdf(df_col):
    """
    return a pandas frame describing the pdf and cdf of each category
    :param df_col: a supposedly categorical column from a pandas df
    :return: a pandas dataframe with
        column1: each category from df_col
        column2: pdf of the cateogry (in descending order)
        column3: cdf of the category
    """
    pdf_df = pd.DataFrame(df_col.value_counts(1)).reset_index()
    pdf_df.columns = [df_col.name, 'pdf']
    pdf_df['cdf'] = np.cumsum(pdf_df['pdf'])
    return pdf_df


def na_agg(df, cols=[], missing_only=False, plot=True):
    """[summary]
    
    Arguments:
        df {pandas dataframe} 
    
    Keyword Arguments:
        cols {list} -- [description] (default: {[]})
        missing_only {bool} -- [description] (default: {False})
        plot {bool} -- [description] (default: {True})
    
    Returns:
        pandas dataframe -- showing each column, its missing value counts and missing value percentage
    """
    if len(cols) == 0:
        cols = df.columns
    # Capture the necessary data
    count = []

    for variable in cols:
        length = df[variable].count()
        count.append(length)

    count_pct = np.round(100 * pd.Series(count) / len(df), 2)
    count = pd.Series(count)

    missing = pd.DataFrame()
    missing['variables'] = cols
    missing['count'] = len(df) - count
    missing['count_pct'] = 100 - count_pct

    if missing_only:
        missing = missing[missing['count'] > 0]

    missing.sort_values(by=['count_pct'], inplace=True)
    missing_train = np.array(missing['variables'])

    if plot:
        # Plot number of available data per variable
        plt.subplots(figsize=(15, 6))

        # Plots missing data in percentage
        plt.subplot(1, 2, 1)
        plt.barh(missing['variables'], missing['count_pct'])
        plt.title('Count of missing training data in percent', fontsize=15)

        # Plots total row number of missing data
        plt.subplot(1, 2, 2)
        plt.barh(missing['variables'], missing['count'])
        plt.title('Count of missing training data as total records', fontsize=15)

        plt.show()
    return missing
