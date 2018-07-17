from csv import reader as csv_reader
from itertools import chain

__all__ = ["csv_open"]


def csv_open(file, expected_columns):
    """
    Yields rows of csv file as dictionaries

    Parameters:
        file - Path, or file-like object, of the CSV file to use
        expected_columns - Columns of the csv file
            If the first row of the CSV file are these labels, take the columns in that order
            Otherwise, take the columns in the order given by expected_columns
    """

    if isinstance(file, str):
        with open(file) as f:
            yield from csv_open(f, expected_columns=expected_columns)
            return

    expected_columns = tuple(expected_columns)

    csv_iter = csv_reader(file)

    first_row = next(csv_iter)

    if set(first_row) == set(expected_columns):
        columns = first_row
    else:
        columns = expected_columns
        csv_iter = chain([first_row], csv_iter)

    for row in csv_iter:
        if len(row) < len(columns):
            raise IndexError("Too few columns in row {!r}".format(row))

        yield dict(zip(columns, row))
