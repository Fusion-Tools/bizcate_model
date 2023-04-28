# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule
from callbacks import Scaler


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
            BIZCATE_CODE=_.SUB_CODE,
        )
        >> filter(
            ~_.OPTION.isna(),
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.BIZCATE_CODE,
            _.OPTION,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.PERCENT_YES_BODIES,
        )
        >> collect()
    )

    return a002_q23bodies


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
        process_std=0.020,
    )


def corr_kf_module(metric_col):
    return BizcateCorrelationKFModule(
        metric_col=metric_col,
        output_col_prefix=metric_col + "_CORR",
        sample_size_col="ASK_COUNT",
        process_std=0.020,
    )


# %%
# fmt: off
runner = Runner(
    callbacks=[
        # Scaler(),
        LogitTransform(),
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

outputs = []

# %%
a002_q23bodies = load_q23bodies()

# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a002_national = (
    a002_q23bodies
    >> filter(_.CUT_ID == 1)
    >> rename(PERCENT_YES_BODIES_NATIONAL=_.PERCENT_YES_BODIES)
)

a002_national_dl = dataloader(a002_national)
a002_national_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_BODIES_NATIONAL")
a002_national_kf_module_corr = corr_kf_module("PERCENT_YES_BODIES_NATIONAL")

a002_national_filtered = runner.run(
    models=[
        a002_national_kf_module_no_corr,
        a002_national_kf_module_corr,
    ],
    dataloaders=a002_national_dl,
)

# %%
a002_national_filtered_renamed = (
    a002_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a002_national_filtered.columns}
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

outputs.append(a002_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a002_demo = (
    a002_q23bodies
    >> filter(_.CUT_ID != 1)
    >> rename(PERCENT_YES_BODIES_DEMO=_.PERCENT_YES_BODIES)
)

# %% calculate demo cuts deltas
a002_demo_delta = (
    inner_join(
        a002_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT
        ),  # fmt: skip
        a002_demo,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        PERCENT_YES_BODIES_DELTA=_.PERCENT_YES_BODIES_NATIONAL - _.PERCENT_YES_BODIES_DEMO  # fmt: skip
    )
    >> select(~_.PERCENT_YES_BODIES_NATIONAL)
)


# %% filter demo cuts deltas
a002_demo_delta_dl = dataloader(a002_demo_delta)
a002_demo_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_BODIES_DELTA")
a002_demo_kf_module_corr = corr_kf_module("PERCENT_YES_BODIES_DELTA")

a002_demo_delta_filtered = runner.run(
    models=[a002_demo_kf_module_no_corr, a002_demo_kf_module_corr],
    dataloaders=a002_demo_delta_dl,
)

# %% apply national
a002_demo_filtered = (
    inner_join(
        a002_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT),
        a002_demo_delta_filtered,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    # fmt: off
    >> mutate(
        PERCENT_YES_BODIES_DEMO_NO_CORR_KF=_.PERCENT_YES_BODIES_NATIONAL_NO_CORR_KF - _.PERCENT_YES_BODIES_DELTA_NO_CORR_KF,
        PERCENT_YES_BODIES_DEMO_NO_CORR_RTS=_.PERCENT_YES_BODIES_NATIONAL_NO_CORR_RTS - _.PERCENT_YES_BODIES_DELTA_NO_CORR_RTS,
        PERCENT_YES_BODIES_DEMO_CORR_KF=_.PERCENT_YES_BODIES_NATIONAL_CORR_KF - _.PERCENT_YES_BODIES_DELTA_CORR_KF,
        PERCENT_YES_BODIES_DEMO_CORR_RTS=_.PERCENT_YES_BODIES_NATIONAL_CORR_RTS - _.PERCENT_YES_BODIES_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.PERCENT_YES_BODIES_DELTA,
        ~_.PERCENT_YES_BODIES_DELTA_NO_CORR_KF,
        ~_.PERCENT_YES_BODIES_DELTA_NO_CORR_RTS,
        ~_.PERCENT_YES_BODIES_DELTA_CORR_KF,
        ~_.PERCENT_YES_BODIES_DELTA_CORR_RTS,
        ~_.PERCENT_YES_BODIES_NATIONAL,
        ~_.PERCENT_YES_BODIES_NATIONAL_NO_CORR_KF,
        ~_.PERCENT_YES_BODIES_NATIONAL_NO_CORR_RTS,
        ~_.PERCENT_YES_BODIES_NATIONAL_CORR_KF,
        ~_.PERCENT_YES_BODIES_NATIONAL_CORR_RTS,
    )
)

# %%
a002_demo_filtered_renamed = (
    a002_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a002_demo_filtered.columns})
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

outputs.append(a002_demo_filtered_renamed)

# %%
a002_filtered = pd.concat(outputs, ignore_index=True)

# %% upload
fdb.upload(
    df=a002_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="BIZCATE_M002_Q23BODIES",
    if_exists="replace",
)

# %% test
# (
#     a002_filtered
#     >> filter(
#         _.CUT_ID == 1,
#         _.OPTION == 1,
#         _.BIZCATE_CODE == 197,
#     )
# ).plot(
#     x = "MONTH_YEAR",
#     y = [
#         "PERCENT_YES_BODIES",
#         "PERCENT_YES_BODIES_NO_CORR_KF",
#         "PERCENT_YES_BODIES_NO_CORR_RTS",
#         "PERCENT_YES_BODIES_CORR_KF",
#         "PERCENT_YES_BODIES_CORR_RTS",
#     ]
# ).legend(loc='best')

# %%
