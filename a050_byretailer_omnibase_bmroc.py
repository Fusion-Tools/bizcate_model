# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *
from siuba.experimental.pivot import pivot_longer
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule
from utils import logit, inv_logit


# %%
def load_bmroc(
    database="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    table="A050_BYRETAILER_OMNIBASE_BMROC",
    cut_ids=None,
    logit_transform=True,
):
    a050_bmroc = (
        fdb[database][schema][table](lazy=True)
        # TODO: temp fix
        >> rename(
            MONTH_NUM=_.MONTH,
            BIZCATE_CODE=_.SUB_CODE,
        )
        >> filter(
            ~_.CUT_ID.isna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
            _.IMPUTED == 0, # TODO: ?
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.BIZCATE_CODE,
            _.RETAILER_CODE,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT_SPEND,
            _.TA == _.PERCENT_SPEND_FINAL_TA_AND_SHARE_ADJUSTED_11,
            _.THINK == _.PERCENT_SPEND_FINAL_TA_AND_SHARE_ADJUSTED_12,
            _.CONSIDER == _.PERCENT_SPEND_FINAL_TA_AND_SHARE_ADJUSTED_13,
            _.VISIT == _.PERCENT_SPEND_FINAL_TA_AND_SHARE_ADJUSTED_14,
            _.SHARE == _.PERCENT_SPEND_FINAL_TA_AND_SHARE_ADJUSTED_16,
        )
        >> pivot_longer(
            _["TA", "THINK", "CONSIDER", "VISIT", "SHARE"],
            names_to = "METRIC",
            values_to="SCORE",
        )
        >> collect()
    )

    if logit_transform:
        a050_bmroc["SCORE"] = logit(a050_bmroc["SCORE"])

    return a050_bmroc

# %%
def dataloader(table):
    return DataLoader(
        table=table,
        id_cols=["CUT_ID", "RETAILER_CODE", "METRIC"],
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
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

outputs = []

# %%
a050_bmroc = load_bmroc()

# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a050_national = (
    a050_bmroc
    >> filter(_.CUT_ID == 1)
    >> rename(SCORE_NATIONAL=_.SCORE)
)

a050_national_dl = dataloader(a050_national)
a050_national_kf_module_no_corr = no_corr_kf_module("SCORE_NATIONAL")
a050_national_kf_module_corr = corr_kf_module("SCORE_NATIONAL")

a050_national_filtered = runner.run(
    models=[
        a050_national_kf_module_no_corr,
        a050_national_kf_module_corr,
    ],
    dataloaders=a050_national_dl,
)

# %%
a050_national_filtered_renamed = (
    a050_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a050_national_filtered.columns}
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

outputs.append(a050_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a050_demo = (
    a050_bmroc
    >> filter(_.CUT_ID != 1)
    >> rename(SCORE_DEMO=_.SCORE)
)

# %% calculate demo cuts deltas
a050_demo_delta = (
    inner_join(
        a050_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT_SPEND
        ),  # fmt: skip
        a050_demo,
        on=[
            "RETAILER_CODE", 
            "METRIC",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        SCORE_DELTA=_.SCORE_NATIONAL - _.SCORE_DEMO  # fmt: skip
    )
    >> select(~_.SCORE_NATIONAL)
)


# %% filter demo cuts deltas
a050_demo_delta_dl = dataloader(a050_demo_delta)
a050_demo_kf_module_no_corr = no_corr_kf_module("SCORE_DELTA")
a050_demo_kf_module_corr = corr_kf_module("SCORE_DELTA")

a050_demo_delta_filtered = runner.run(
    models=[a050_demo_kf_module_no_corr, a050_demo_kf_module_corr],
    dataloaders=a050_demo_delta_dl,
)

# %% apply national
a050_demo_filtered = (
    inner_join(
        a050_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT_SPEND),
        a050_demo_delta_filtered,
        on=[
            "RETAILER_CODE", 
            "METRIC",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> select(~_.endswith("_KF"))
    # fmt: off
    >> mutate(
        # SCORE_DEMO_NO_CORR_KF=_.SCORE_NATIONAL_NO_CORR_KF - _.SCORE_DELTA_NO_CORR_KF,
        SCORE_DEMO_NO_CORR_RTS=_.SCORE_NATIONAL_NO_CORR_RTS - _.SCORE_DELTA_NO_CORR_RTS,
        # SCORE_DEMO_CORR_KF=_.SCORE_NATIONAL_CORR_KF - _.SCORE_DELTA_CORR_KF,
        SCORE_DEMO_CORR_RTS=_.SCORE_NATIONAL_CORR_RTS - _.SCORE_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.SCORE_DELTA,
        # ~_.SCORE_DELTA_NO_CORR_KF,
        ~_.SCORE_DELTA_NO_CORR_RTS,
        # ~_.SCORE_DELTA_CORR_KF,
        ~_.SCORE_DELTA_CORR_RTS,
        ~_.SCORE_NATIONAL,
        # ~_.SCORE_NATIONAL_NO_CORR_KF,
        ~_.SCORE_NATIONAL_NO_CORR_RTS,
        # ~_.SCORE_NATIONAL_CORR_KF,
        ~_.SCORE_NATIONAL_CORR_RTS,
    )
)

# %%
a050_demo_filtered_renamed = (
    a050_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a050_demo_filtered.columns})
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

outputs.append(a050_demo_filtered_renamed)

# %%
a050_filtered = pd.concat(outputs, ignore_index=True)

# %%
a050_filtered["SCORE"] = inv_logit(a050_filtered["SCORE"])
a050_filtered["SCORE_NO_CORR_RTS"] = inv_logit(a050_filtered["SCORE_NO_CORR_RTS"])
a050_filtered["SCORE_CORR_RTS"] = inv_logit(a050_filtered["SCORE_CORR_RTS"])

# %% upload
fdb.upload(
    df=a050_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="BIZCATE_M050_BMROC",
    if_exists="replace",
)

# %% test
(
    a050_filtered
    >> filter(
        _.CUT_ID == 1,
        _.RETAILER_CODE == 115,
        _.BIZCATE_CODE == 111,
        _.METRIC == "THINK"
    )
).plot(
    x = "MONTH_YEAR",
    y = [
        "SCORE",
        "SCORE_NO_CORR_RTS",
        "SCORE_CORR_RTS",
    ]
).legend(loc='best')

# %%
