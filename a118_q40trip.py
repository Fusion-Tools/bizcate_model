# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule

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


def dataloader(table):
    return DataLoader(
        table=table,
        id_cols=["CUT_ID", "OPTION"],
        date_col="MONTH_YEAR",
        var_cols=["BIZCATE_CODE"],
    )


def no_corr_kf_module(metric_col):
    return NoCorrelationKFModule(
        metric_col=metric_col,
        output_col_prefix=metric_col + "_NO_CORR",
        sample_size_col="ASK_COUNT",
        process_std=0.015,
    )


def corr_kf_module(metric_col):
    return BizcateCorrelationKFModule(
        metric_col=metric_col,
        output_col_prefix=metric_col + "_CORR",
        sample_size_col="ASK_COUNT",
        process_std=0.015,
    )


# %%
# fmt: off
runner = Runner(
    callbacks=[
        LogitTransform(),
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

outputs = []

# %%
a003_q23spend = load_q23spend()

# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a003_national = (
    a003_q23spend
    >> filter(_.CUT_ID == 1)
    >> rename(PERCENT_YES_SPEND_NATIONAL=_.PERCENT_YES_SPEND)
)

a003_national_dl = dataloader(a003_national)
a003_national_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_SPEND_NATIONAL")
a003_national_kf_module_corr = corr_kf_module("PERCENT_YES_SPEND_NATIONAL")

a003_national_filtered = runner.run(
    models=[
        a003_national_kf_module_no_corr,
        a003_national_kf_module_corr,
    ],
    dataloaders=a003_national_dl,
)

# %%
a003_national_filtered_renamed = (
    a003_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a003_national_filtered.columns}
    )
    >> mutate(
        across(
            _[_.endswith("_KF"), _.endswith("_RTS")],
            if_else(Fx < 0, 0, Fx),
        ),
    )
    >> mutate(
        across(
            _[_.endswith("_KF"), _.endswith("_RTS")],
            if_else(Fx > 1, 1, Fx),
        ),
    )
)

outputs.append(a003_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a003_demo = (
    a003_q23spend
    >> filter(_.CUT_ID != 1)
    >> rename(PERCENT_YES_SPEND_DEMO=_.PERCENT_YES_SPEND)
)

# %% calculate demo cuts deltas
a003_demo_delta = (
    inner_join(
        a003_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT
        ),  # fmt: skip
        a003_demo,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        PERCENT_YES_SPEND_DELTA=_.PERCENT_YES_SPEND_NATIONAL - _.PERCENT_YES_SPEND_DEMO  # fmt: skip
    )
    >> select(~_.PERCENT_YES_SPEND_NATIONAL)
)


# %% filter demo cuts deltas
a003_demo_delta_dl = dataloader(a003_demo_delta)
a003_demo_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_SPEND_DELTA")
a003_demo_kf_module_corr = corr_kf_module("PERCENT_YES_SPEND_DELTA")

a003_demo_delta_filtered = runner.run(
    models=[a003_demo_kf_module_no_corr, a003_demo_kf_module_corr],
    dataloaders=a003_demo_delta_dl,
)

# %% apply national
a003_demo_filtered = (
    inner_join(
        a003_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT),
        a003_demo_delta_filtered,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    # fmt: off
    >> mutate(
        PERCENT_YES_SPEND_DEMO_NO_CORR_KF=_.PERCENT_YES_SPEND_NATIONAL_NO_CORR_KF - _.PERCENT_YES_SPEND_DELTA_NO_CORR_KF,
        PERCENT_YES_SPEND_DEMO_NO_CORR_RTS=_.PERCENT_YES_SPEND_NATIONAL_NO_CORR_RTS - _.PERCENT_YES_SPEND_DELTA_NO_CORR_RTS,
        PERCENT_YES_SPEND_DEMO_CORR_KF=_.PERCENT_YES_SPEND_NATIONAL_CORR_KF - _.PERCENT_YES_SPEND_DELTA_CORR_KF,
        PERCENT_YES_SPEND_DEMO_CORR_RTS=_.PERCENT_YES_SPEND_NATIONAL_CORR_RTS - _.PERCENT_YES_SPEND_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.PERCENT_YES_SPEND_DELTA,
        ~_.PERCENT_YES_SPEND_DELTA_NO_CORR_KF,
        ~_.PERCENT_YES_SPEND_DELTA_NO_CORR_RTS,
        ~_.PERCENT_YES_SPEND_DELTA_CORR_KF,
        ~_.PERCENT_YES_SPEND_DELTA_CORR_RTS,
        ~_.PERCENT_YES_SPEND_NATIONAL,
        ~_.PERCENT_YES_SPEND_NATIONAL_NO_CORR_KF,
        ~_.PERCENT_YES_SPEND_NATIONAL_NO_CORR_RTS,
        ~_.PERCENT_YES_SPEND_NATIONAL_CORR_KF,
        ~_.PERCENT_YES_SPEND_NATIONAL_CORR_RTS,
    )
)

# %%
a003_demo_filtered_renamed = (
    a003_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a003_demo_filtered.columns})
    >> mutate(
        across(
            _[_.endswith("_KF"), _.endswith("_RTS")],
            if_else(Fx < 0, 0, Fx),
        ),
    )
    >> mutate(
        across(
            _[_.endswith("_KF"), _.endswith("_RTS")],
            if_else(Fx > 1, 1, Fx),
        ),
    )
)

outputs.append(a003_demo_filtered_renamed)

# %%
a003_filtered = pd.concat(outputs, ignore_index=True)

# %% upload
fdb.upload(
    df=a003_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="BIZCATE_M003_Q23SPEND",
    if_exists="replace",
)

# %% test
(
    a003_filtered
    >> filter(
        _.CUT_ID == 1,
        _.OPTION == 2,
        _.BIZCATE_CODE == 112,
    )
).plot(
    x = "MONTH_YEAR",
    y = [
        "PERCENT_YES_SPEND",
        "PERCENT_YES_SPEND_NO_CORR_RTS",
        "PERCENT_YES_SPEND_CORR_RTS",
    ]
).legend(loc='best')

# %%
