# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *


# %%
def load_q40trip(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A118_Q40TRIP",
    cut_ids=None,
):
    """Q42a - Online Research purpose - ecom"""

    a118_q40trip = (
        fdb[database][schema][table](lazy=True)
        >> filter(
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.SUB_CODE,  # FIXME: ?
            _.RETAILER_CODE,
            _.OPTION,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.PERCENT_YES,
        )
        >> collect()
    )

    return a118_q40trip


# %%
a118_q40trip = load_q40trip()


# %%
