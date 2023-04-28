# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule

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
            _.BIZCATE_CODE,
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
a100_q42research_purpose_ecom = load_q42research_purpose_ecom()

# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a100_national = (
    a100_q42research_purpose_ecom
    >> filter(_.CUT_ID == 1)
    >> rename(PERCENT_YES_NATIONAL=_.PERCENT_YES)
)

a100_national_dl = dataloader(a100_national)
a100_national_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_NATIONAL")
a100_national_kf_module_corr = corr_kf_module("PERCENT_YES_NATIONAL")

a100_national_filtered = runner.run(
    models=[
        a100_national_kf_module_no_corr,
        a100_national_kf_module_corr,
    ],
    dataloaders=a100_national_dl,
)

# %%
a100_national_filtered_renamed = (
    a100_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a100_national_filtered.columns}
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

outputs.append(a100_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a100_demo = (
    a100_q42research_purpose_ecom
    >> filter(_.CUT_ID != 1)
    >> rename(PERCENT_YES_DEMO=_.PERCENT_YES)
)

# %% calculate demo cuts deltas
a100_demo_delta = (
    inner_join(
        a100_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT
        ),  # fmt: skip
        a100_demo,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        PERCENT_YES_DELTA=_.PERCENT_YES_NATIONAL - _.PERCENT_YES_DEMO  # fmt: skip
    )
    >> select(~_.PERCENT_YES_NATIONAL)
)


# %% filter demo cuts deltas
a100_demo_delta_dl = dataloader(a100_demo_delta)
a100_demo_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_DELTA")
a100_demo_kf_module_corr = corr_kf_module("PERCENT_YES_DELTA")

a100_demo_delta_filtered = runner.run(
    models=[a100_demo_kf_module_no_corr, a100_demo_kf_module_corr],
    dataloaders=a100_demo_delta_dl,
)

# %% apply national
a100_demo_filtered = (
    inner_join(
        a100_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT),
        a100_demo_delta_filtered,
        on=[
            "OPTION",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    # fmt: off
    >> mutate(
        PERCENT_YES_DEMO_NO_CORR_KF=_.PERCENT_YES_NATIONAL_NO_CORR_KF - _.PERCENT_YES_DELTA_NO_CORR_KF,
        PERCENT_YES_DEMO_NO_CORR_RTS=_.PERCENT_YES_NATIONAL_NO_CORR_RTS - _.PERCENT_YES_DELTA_NO_CORR_RTS,
        PERCENT_YES_DEMO_CORR_KF=_.PERCENT_YES_NATIONAL_CORR_KF - _.PERCENT_YES_DELTA_CORR_KF,
        PERCENT_YES_DEMO_CORR_RTS=_.PERCENT_YES_NATIONAL_CORR_RTS - _.PERCENT_YES_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.PERCENT_YES_DELTA,
        ~_.PERCENT_YES_DELTA_NO_CORR_KF,
        ~_.PERCENT_YES_DELTA_NO_CORR_RTS,
        ~_.PERCENT_YES_DELTA_CORR_KF,
        ~_.PERCENT_YES_DELTA_CORR_RTS,
        ~_.PERCENT_YES_NATIONAL,
        ~_.PERCENT_YES_NATIONAL_NO_CORR_KF,
        ~_.PERCENT_YES_NATIONAL_NO_CORR_RTS,
        ~_.PERCENT_YES_NATIONAL_CORR_KF,
        ~_.PERCENT_YES_NATIONAL_CORR_RTS,
    )
)

# %%
a100_demo_filtered_renamed = (
    a100_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a100_demo_filtered.columns})
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

outputs.append(a100_demo_filtered_renamed)

# %%
a100_filtered = pd.concat(outputs, ignore_index=True)

# %% upload
fdb.upload(
    df=a100_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="BIZCATE_M100_Q42RESEARCH_PURPOSE_ECOM",
    if_exists="replace",
)

# %% test
# (
#     a100_filtered
#     >> filter(
#         _.CUT_ID == 1,
#         _.OPTION == 2,
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