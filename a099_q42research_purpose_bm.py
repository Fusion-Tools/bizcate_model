# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *


# %%
def load_q42research_purpose_bm(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A099_Q42RESEARCH_PURPOSE_BM",
    cut_ids=None,
):
    """Q42a - Online Research purpose - brick"""

    a099_q42research_purpose_bm = (
        fdb[database][schema][table](lazy=True)
        >> filter(
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.SUB_CODE,  # FIXME: ?
            _.OPTION,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.PERCENT_YES,
        )
        >> collect()
    )

    return a099_q42research_purpose_bm


# %%
a099_q42research_purpose_bm = load_q42research_purpose_bm()


# %%
