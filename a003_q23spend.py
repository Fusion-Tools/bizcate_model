# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *


# %%
def load_q23spend(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A003_Q23SPEND",
    cut_ids=None,
):
    a002_q23spend = (
        fdb[database][schema][table](lazy=True)
        # TODO: temp fix
        >> rename(
            MONTH_NUM=_.MONTH,
        )
        >> filter(
            ~_.OPTION.isna(),
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.CATEGORY_CODE,
            _.OPTION,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.PERCENT_YES_SPEND,
        )
        >> collect()
    )

    return a002_q23spend


# %%
a002_q23spend = load_q23spend()


# %%
