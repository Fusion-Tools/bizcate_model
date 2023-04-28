from db import fdb, month_mapping
import pandas as pd
from siuba import *
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule


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
            BIZCATE_CODE=_.SUB_CODE
        )
        >> filter(
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.BIZCATE_CODE,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.AVG_SPEND,
        )
        >> collect()
    )

    return a018_avgspend_bm

   
# %%
a018 = load_avgspend_bm()




# %%
bizcate_mapping = (
    fdb.FUSEDDATA.LEVER_JSTEP.LOOKUP_BIZCATE_SUBCATE_QUOTA(lazy=True)
    >> filter(
        _.BIZCATE_CODE.notna(),
    )
    >> distinct(
        _.BIZCATE_CODE, 
        _.BIZCATE
    )
    >> collect()
)

# %%
(
    bizcate_mapping 
    >> filter(
        _.BIZCATE_CODE== 449
    )
)

# %%
(
    a018
    >> filter(
        _.CUT_ID == 1
    )
    >> group_by(
        _.BIZCATE_CODE,
    )
    >> summarize(
        MEAN_AVG_SPEND = _.AVG_SPEND.mean(),
        MEAN_ASK_COUNT = _.ASK_COUNT.mean(),
    )
    >> arrange(
        _.MEAN_ASK_COUNT
    )
)
# %%
