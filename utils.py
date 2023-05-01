# %%
import warnings

warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
from siuba import *
from siuba.dply.vector import *
from siuba.siu.dispatchers import verb_dispatch
from siuba.sql import LazyTbl
import io
from contextlib import redirect_stdout
from datetime import date
from dateutil.relativedelta import relativedelta



# %%
def snowflake_tbl_colnames(lazy_tbl):
    cols = (lazy_tbl >> head(1) >> collect()).columns.values.tolist()
    return cols


@verb_dispatch(LazyTbl)
def rename_cols_toupper(__data):
    cols = snowflake_tbl_colnames(__data)
    new_lazy_tbl = __data >> rename(**{str(col).upper(): col for col in cols})
    return new_lazy_tbl  # FIXME: variable name


@verb_dispatch(LazyTbl)
def get_distinct(lazy_tbl, col):
    distinct_vals = (
        lazy_tbl >> rename_cols_toupper() >> distinct(col) >> arrange() >> collect()
    )[col].tolist()
    return distinct_vals


@get_distinct.register(pd.DataFrame)
def _get_distinct_df(df, col):
    distinct_vals = df[col].unique().tolist()
    return sorted(distinct_vals)


@verb_dispatch(pd.DataFrame)
def get_distinct_level(df, level):
    distinct_vals = df.index.unique(level=level).tolist()
    return sorted(distinct_vals)


@verb_dispatch(pd.DataFrame)
def collect_col_as_list(__data, col):
    col_as_list = (__data >> select(col))[col].tolist()
    return col_as_list


@collect_col_as_list.register(LazyTbl)
def _collect_col_as_list_lazy_tbl(lazy_tbl, col):
    col_as_list = (lazy_tbl >> select(col) >> collect())[col].tolist()
    return col_as_list


@verb_dispatch(pd.DataFrame)
def reset_index(__data, drop=True):
    return __data.reset_index(drop=drop)


def get_query_string(lazy_tbl):
    f = io.StringIO()  # create a memory file object

    with redirect_stdout(f):  # redirect stdout to f
        tmp = lazy_tbl >> show_query(simplify=True)  # call the function that prints to stdout

    return f.getvalue()  # get the captured output as a string


def first_day_of_previous_month():
    today = date.today()
    last_month_start = today.replace(day=1) - relativedelta(months=1)
    return last_month_start

def logit(x, min=0.00000001, max=0.99999999):
    """Convert to logit space"""

    p = np.array(x)
    p[p < min] = min
    p[p > max] = max
    return np.log(p / (1 - p))


def inv_logit(x):
    """Convert from logit space"""

    p = np.exp(x) / (1 + np.exp(x))
    p[np.isnan(p) & ~np.isnan(x)] = 1
    return p