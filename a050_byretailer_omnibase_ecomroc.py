# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *


# %%
def load_ecomroc(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A050_BYRETAILER_OMNIBASE_ECOMROC",
    cut_ids=None,
):
    a050_ecomroc = (
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
            _.RETAILER_CODE,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT_SPEND,
            _.IMPUTED,
            _.startswith("PERCENT_"),
            _.startswith("TA_"),
        )
        >> collect()
    )

    return a050_ecomroc


# %%
a050_ecomroc = load_ecomroc()


# %%
