import pandas as pd
from pandas.api import types

from typing import Callable


class Function:
    _string: str = ""
    _func: Callable[[float], float] = lambda x: 0

    def __init__(self, s, f):
        self._string = s
        self._func = f

    def at(self, x: float) -> float:
        return self._func(x)

    def __str__(self):
        return "function: (" + self._string + ")"


class TableFunction:
    _table: pd.DataFrame = None     # | x | y |

    def __init__(self, t: pd.DataFrame):
        columns = t.columns
        assert len(columns) == 2 and columns[0] == 'x' and columns[1] == 'y', "Must contains only (x,y) cols"
        assert all(t.notnull()), "Must contains only non null values"
        assert types.is_numeric_dtype(t['x']) and types.is_numeric_dtype(t['y']), "Must have numeric values"
        assert len(t['x']) == len(t['x'].drop_duplicates()), "All x values must be unique"
        self._table = t

    def table(self) -> pd.DataFrame:
        return self._table

    def x_values(self) -> pd.Series:
        return self._table['x']

    def y_values(self) -> pd.Series:
        return self._table['y']
