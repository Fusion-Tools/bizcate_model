# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *


# %%
def load_q23bodies(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A002_Q23BODIES",
    cut_ids=None,
):
    a002_q23bodies = (
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
            _.PERCENT_YES_BODIES,
        )
        >> collect()
    )

    return a002_q23bodies


# %%
a002_q23bodies = load_q23bodies()

# %%
