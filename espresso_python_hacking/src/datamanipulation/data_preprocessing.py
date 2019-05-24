import numpy as np


def identify_quant_cols(
        # TODO: add any arguments here
):
    # TODO: write code that will return a list of the quantitative columns in a dataframe
    pass


def make_col_positive(data):
    # TODO: Add transformations here to make an entire dataframe column positive.
    minval = data.min()
    data = data + abs(minval)
    return data


def log_transform(data):
    # TODO: Add any code here to log transform an entire column.
    return np.log(data)



