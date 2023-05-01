# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule
from utils import logit, inv_logit

# %%
def load_q40trip(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A118_Q40TRIP",
    cut_ids=None,
    logit_transform=True,
):
    """Q42a - Online Research purpose - ecom"""

    a118_q40trip = (
        fdb[database][schema][table](lazy=True)
        >> rename(
            BIZCATE_CODE = _.SUB_CODE, 
        )
        >> filter(
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.BIZCATE_CODE,  # FIXME: ?
            _.RETAILER_CODE,
            _.OPTION,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.PERCENT_YES,
        )
        >> collect()
    )

    if logit_transform:
        a118_q40trip["PERCENT_YES"] = logit(a118_q40trip["PERCENT_YES"])


    return a118_q40trip


# %%
def dataloader(table):
    return DataLoader(
        table=table,
        id_cols=["CUT_ID", "OPTION", "RETAILER_CODE"],
        date_col="MONTH_YEAR",
        var_cols=["BIZCATE_CODE"],
    )


def no_corr_kf_module(metric_col):
    return NoCorrelationKFModule(
        metric_col=metric_col,
        output_col_prefix=metric_col + "_NO_CORR",
        sample_size_col="ASK_COUNT",
        process_std=0.020,
    )


def corr_kf_module(metric_col):
    return BizcateCorrelationKFModule(
        metric_col=metric_col,
        output_col_prefix=metric_col + "_CORR",
        sample_size_col="ASK_COUNT",
        process_std=0.020
    )


# %%
# fmt: off
runner = Runner(
    callbacks=[
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

outputs = []

# %%
a118_q40trip = load_q40trip()

# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a118_national = (
    a118_q40trip
    >> filter(_.CUT_ID == 1)
    >> rename(PERCENT_YES_NATIONAL=_.PERCENT_YES)
)

a118_national_dl = dataloader(a118_national)
a118_national_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_NATIONAL")
a118_national_kf_module_corr = corr_kf_module("PERCENT_YES_NATIONAL")

a118_national_filtered = runner.run(
    models=[
        a118_national_kf_module_no_corr,
        a118_national_kf_module_corr,
    ],
    dataloaders=a118_national_dl,
)

# %%
a118_national_filtered_renamed = (
    a118_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a118_national_filtered.columns}
    )
    >> select(~_.endswith("_KF"))
    # >> mutate(
    #     across(
    #         _[_.endswith("_KF"), _.endswith("_RTS")],
    #         if_else(Fx < 0, 0, Fx),
    #     ),
    # )
    # >> mutate(
    #     across(
    #         _[_.endswith("_KF"), _.endswith("_RTS")],
    #         if_else(Fx > 1, 1, Fx),
    #     ),
    # )
)

outputs.append(a118_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a118_demo = (
    a118_q40trip
    >> filter(_.CUT_ID != 1)
    >> rename(PERCENT_YES_DEMO=_.PERCENT_YES)
)

# %% calculate demo cuts deltas
a118_demo_delta = (
    inner_join(
        a118_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT
        ),  # fmt: skip
        a118_demo,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "RETAILER_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        PERCENT_YES_DELTA=_.PERCENT_YES_NATIONAL - _.PERCENT_YES_DEMO  # fmt: skip
    )
    >> select(~_.PERCENT_YES_NATIONAL)
)


# %% filter demo cuts deltas
a118_demo_delta_dl = dataloader(a118_demo_delta)
a118_demo_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_DELTA")
a118_demo_kf_module_corr = corr_kf_module("PERCENT_YES_DELTA")

a118_demo_delta_filtered = runner.run(
    models=[a118_demo_kf_module_no_corr, a118_demo_kf_module_corr],
    dataloaders=a118_demo_delta_dl,
)

# %% apply national
a118_demo_filtered = (
    inner_join(
        a118_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT),
        a118_demo_delta_filtered,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "RETAILER_CODE",
            "MONTH_YEAR",
        ],
    )
    >> select(~_.endswith("_KF"))
    # fmt: off
    >> mutate(
        # PERCENT_YES_DEMO_NO_CORR_KF=_.PERCENT_YES_NATIONAL_NO_CORR_KF - _.PERCENT_YES_DELTA_NO_CORR_KF,
        PERCENT_YES_DEMO_NO_CORR_RTS=_.PERCENT_YES_NATIONAL_NO_CORR_RTS - _.PERCENT_YES_DELTA_NO_CORR_RTS,
        # PERCENT_YES_DEMO_CORR_KF=_.PERCENT_YES_NATIONAL_CORR_KF - _.PERCENT_YES_DELTA_CORR_KF,
        PERCENT_YES_DEMO_CORR_RTS=_.PERCENT_YES_NATIONAL_CORR_RTS - _.PERCENT_YES_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.PERCENT_YES_DELTA,
        # ~_.PERCENT_YES_DELTA_NO_CORR_KF,
        ~_.PERCENT_YES_DELTA_NO_CORR_RTS,
        # ~_.PERCENT_YES_DELTA_CORR_KF,
        ~_.PERCENT_YES_DELTA_CORR_RTS,
        ~_.PERCENT_YES_NATIONAL,
        # ~_.PERCENT_YES_NATIONAL_NO_CORR_KF,
        ~_.PERCENT_YES_NATIONAL_NO_CORR_RTS,
        # ~_.PERCENT_YES_NATIONAL_CORR_KF,
        ~_.PERCENT_YES_NATIONAL_CORR_RTS,
    )
)

# %%
a118_demo_filtered_renamed = (
    a118_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a118_demo_filtered.columns})
    # >> mutate(
    #     across(
    #         _[_.endswith("_KF"), _.endswith("_RTS")],
    #         if_else(Fx < 0, 0, Fx),
    #     ),
    # )
    # >> mutate(
    #     across(
    #         _[_.endswith("_KF"), _.endswith("_RTS")],
    #         if_else(Fx > 1, 1, Fx),
    #     ),
    # )
)

outputs.append(a118_demo_filtered_renamed)

# %%
a118_filtered = pd.concat(outputs, ignore_index=True)

# %%
a118_filtered["PERCENT_YES"] = inv_logit(a118_filtered["PERCENT_YES"])
a118_filtered["PERCENT_YES_NO_CORR_RTS"] = inv_logit(a118_filtered["PERCENT_YES_NO_CORR_RTS"])
a118_filtered["PERCENT_YES_CORR_RTS"] = inv_logit(a118_filtered["PERCENT_YES_CORR_RTS"])

# %% upload
fdb.upload(
    df=a118_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="BIZCATE_M118_Q40TRIP",
    if_exists="replace",
)

# %% test
# (
#     a118_filtered
#     >> filter(
#         _.CUT_ID == 1,
#         _.OPTION == 1,
#         _.RETAILER_CODE == 45,
#         _.BIZCATE_CODE == 112,
#     )
# ).plot(
#     x = "MONTH_YEAR",
#     y = [
#         "PERCENT_YES",
#         "PERCENT_YES_NO_CORR_RTS",
#         "PERCENT_YES_CORR_RTS",
#     ]
# ).legend(loc='best')

# %%
