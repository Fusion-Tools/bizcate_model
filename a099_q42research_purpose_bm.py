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
def load_q42research_purpose_bm(
    database="L2SURVEY",
    schema="MASPL_ROLLUP",
    table="A098_Q42RESEARCH",
    cut_ids=None,
    logit_transform=True,
):
    """Q42a - Online Research purpose - brick"""

    a098_q42research_purpose_bm = (
        fdb[database][schema][table](lazy=True)
        >> filter(
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.SUB_CODE,
            _.OPTION,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.PERCENT_YES,
        )
        >> collect()
        >> distinct(
            _.CUT_ID,
            _.SUB_CODE,
            _.OPTION,
            _.MONTH_YEAR,
            _keep_all=True
        )
    )

    if logit_transform:
        a098_q42research_purpose_bm["PERCENT_YES"] = logit(a098_q42research_purpose_bm["PERCENT_YES"])

    return a098_q42research_purpose_bm


# %%
def dataloader(table):
    return DataLoader(
        table=table,
        id_cols=["CUT_ID", "OPTION"],
        date_col="MONTH_YEAR",
        var_cols=["SUB_CODE"],
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
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

outputs = []

# %%
a098_q42research_purpose_bm = load_q42research_purpose_bm()

# %%
# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a098_national = (
    a098_q42research_purpose_bm
    >> filter(_.CUT_ID == 1)
    >> rename(PERCENT_YES_NATIONAL=_.PERCENT_YES)
)

a098_national_dl = dataloader(a098_national)
a098_national_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_NATIONAL")
a098_national_kf_module_corr = corr_kf_module("PERCENT_YES_NATIONAL")

a098_national_filtered = runner.run(
    models=[
        a098_national_kf_module_no_corr,
        a098_national_kf_module_corr,
    ],
    dataloaders=a098_national_dl,
)

# %%
a098_national_filtered_renamed = (
    a098_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a098_national_filtered.columns}
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

outputs.append(a098_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a098_demo = (
    a098_q42research_purpose_bm
    >> filter(_.CUT_ID != 1)
    >> rename(PERCENT_YES_DEMO=_.PERCENT_YES)
)

# %% calculate demo cuts deltas
a098_demo_delta = (
    inner_join(
        a098_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT
        ),  # fmt: skip
        a098_demo,
        on=[
            "OPTION",
            "SUB_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        PERCENT_YES_DELTA=_.PERCENT_YES_NATIONAL - _.PERCENT_YES_DEMO  # fmt: skip
    )
    >> select(~_.PERCENT_YES_NATIONAL)
)


# %% filter demo cuts deltas
a098_demo_delta_dl = dataloader(a098_demo_delta)
a098_demo_kf_module_no_corr = no_corr_kf_module("PERCENT_YES_DELTA")
a098_demo_kf_module_corr = corr_kf_module("PERCENT_YES_DELTA")

a098_demo_delta_filtered = runner.run(
    models=[a098_demo_kf_module_no_corr, a098_demo_kf_module_corr],
    dataloaders=a098_demo_delta_dl,
)

# %% apply national
a098_demo_filtered = (
    inner_join(
        a098_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT),
        a098_demo_delta_filtered,
        on=[
            "OPTION",
            "SUB_CODE",
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
a098_demo_filtered_renamed = (
    a098_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a098_demo_filtered.columns})
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

outputs.append(a098_demo_filtered_renamed)

# %%
a098_filtered = pd.concat(outputs, ignore_index=True)

# %%
a098_filtered["PERCENT_YES"] = inv_logit(a098_filtered["PERCENT_YES"])
a098_filtered["PERCENT_YES_NO_CORR_RTS"] = inv_logit(a098_filtered["PERCENT_YES_NO_CORR_RTS"])
a098_filtered["PERCENT_YES_CORR_RTS"] = inv_logit(a098_filtered["PERCENT_YES_CORR_RTS"])

# %% upload
fdb.upload(
    df=a098_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="MASPL_FILTERED_M098_Q42RESEARCH",
    if_exists="replace",
)

# %% test
(
    a098_filtered
    >> filter(
        _.CUT_ID == 1,
        _.OPTION == 2,
        _.SUB_CODE == 362,
    )
).plot(
    x = "MONTH_YEAR",
    y = [
        "PERCENT_YES",
        "PERCENT_YES_NO_CORR_RTS",
        "PERCENT_YES_CORR_RTS",
    ]
).legend(loc='best')

# %%