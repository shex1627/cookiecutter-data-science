import pandas as pd
import numpy as np

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
