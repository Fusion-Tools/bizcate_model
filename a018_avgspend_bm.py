# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *


# %%
def load_avgspend_bm(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A018_AVGSPEND_BM",
    cut_ids=None,
):
    a018_avgspend_bm = (
        fdb[database][schema][table](lazy=True)
        # TODO: temp fix
        >> rename(
            MONTH_NUM=_.MONTH,
        )
        >> filter(
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.CATEGORY_CODE,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.AVG_SPEND,
        )
        >> collect()
    )

    return a018_avgspend_bm


# %%
a018_avgspend_bm = load_avgspend_bm()


# %%
