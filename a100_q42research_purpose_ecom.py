# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *


# %%
def load_q42research_purpose_ecom(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A100_Q42RESEARCH_PURPOSE_ECOM",
    cut_ids=None,
):
    """Q42a - Online Research purpose - ecom"""

    a100_q42research_purpose_ecom = (
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

    return a100_q42research_purpose_ecom


# %%
a100_q42research_purpose_ecom = load_q42research_purpose_ecom()


# %%
