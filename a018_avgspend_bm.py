# %%
from db import fdb, month_mapping
import pandas as pd
from siuba import *
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule
from fusion_kf.callbacks import Scaler
from sklearn.preprocessing import StandardScaler, PowerTransformer


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
def dataloader(table):
    return DataLoader(
        table=table,
        id_cols=["CUT_ID",],
        date_col="MONTH_YEAR",
        var_cols=["BIZCATE_CODE"],
    )


def no_corr_kf_module(metric_col):
    return NoCorrelationKFModule(
        metric_col=metric_col,
        output_col_prefix=metric_col + "_NO_CORR",
        sample_size_col="ASK_COUNT",
        process_std=0.030,
    )


def corr_kf_module(metric_col):
    return BizcateCorrelationKFModule(
        metric_col=metric_col,
        output_col_prefix=metric_col + "_CORR",
        sample_size_col="ASK_COUNT",
        process_std=0.030,
    )


# %%
# fmt: off
runner = Runner(
    callbacks=[
        Scaler(PowerTransformer, method="yeo-johnson"),
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

outputs = []

# %%
a018_avgspend_bm = load_avgspend_bm()


# %%
# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a018_national = (
    a018_avgspend_bm
    >> filter(_.CUT_ID == 1)
    >> rename(AVG_SPEND_NATIONAL=_.AVG_SPEND)
)

a018_national_dl = dataloader(a018_national)
a018_national_kf_module_no_corr = no_corr_kf_module("AVG_SPEND_NATIONAL")
a018_national_kf_module_corr = corr_kf_module("AVG_SPEND_NATIONAL")


# %%
a018_national_filtered = runner.run(
    models=[
        a018_national_kf_module_no_corr,
        a018_national_kf_module_corr,
    ],
    dataloaders=a018_national_dl,
    parallel=True
)

# %%
a018_national_filtered_renamed = (
    a018_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a018_national_filtered.columns}
    )
    >> mutate(
        across(
            _[_.endswith("_KF"), _.endswith("_RTS")],
            if_else(Fx < 0, 0, Fx),
        ),
    )
    # >> mutate(
    #     across(
    #         _[_.endswith("_KF"), _.endswith("_RTS")],
    #         if_else(Fx > 1, 1, Fx),
    #     ),
    # )
)

outputs.append(a018_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a018_demo = (
    a018_avgspend_bm
    >> filter(_.CUT_ID != 1)
    >> rename(AVG_SPEND_DEMO=_.AVG_SPEND)
)

# %% calculate demo cuts deltas
a018_demo_delta = (
    inner_join(
        a018_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT
        ),  # fmt: skip
        a018_demo,
        on=[
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        AVG_SPEND_DELTA=_.AVG_SPEND_NATIONAL - _.AVG_SPEND_DEMO  # fmt: skip
    )
    >> select(~_.AVG_SPEND_NATIONAL)
)


# %% filter demo cuts deltas
a018_demo_delta_dl = dataloader(a018_demo_delta)
a018_demo_kf_module_no_corr = no_corr_kf_module("AVG_SPEND_DELTA")
a018_demo_kf_module_corr = corr_kf_module("AVG_SPEND_DELTA")

a018_demo_delta_filtered = runner.run(
    models=[a018_demo_kf_module_no_corr, a018_demo_kf_module_corr],
    dataloaders=a018_demo_delta_dl,
    parallel=False
)

# %% apply national
a018_demo_filtered = (
    inner_join(
        a018_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT),
        a018_demo_delta_filtered,
        on=[
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    # fmt: off
    >> mutate(
        AVG_SPEND_DEMO_NO_CORR_KF=_.AVG_SPEND_NATIONAL_NO_CORR_KF - _.AVG_SPEND_DELTA_NO_CORR_KF,
        AVG_SPEND_DEMO_NO_CORR_RTS=_.AVG_SPEND_NATIONAL_NO_CORR_RTS - _.AVG_SPEND_DELTA_NO_CORR_RTS,
        AVG_SPEND_DEMO_CORR_KF=_.AVG_SPEND_NATIONAL_CORR_KF - _.AVG_SPEND_DELTA_CORR_KF,
        AVG_SPEND_DEMO_CORR_RTS=_.AVG_SPEND_NATIONAL_CORR_RTS - _.AVG_SPEND_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.AVG_SPEND_DELTA,
        ~_.AVG_SPEND_DELTA_NO_CORR_KF,
        ~_.AVG_SPEND_DELTA_NO_CORR_RTS,
        ~_.AVG_SPEND_DELTA_CORR_KF,
        ~_.AVG_SPEND_DELTA_CORR_RTS,
        ~_.AVG_SPEND_NATIONAL,
        ~_.AVG_SPEND_NATIONAL_NO_CORR_KF,
        ~_.AVG_SPEND_NATIONAL_NO_CORR_RTS,
        ~_.AVG_SPEND_NATIONAL_CORR_KF,
        ~_.AVG_SPEND_NATIONAL_CORR_RTS,
    )
)

# %%
a018_demo_filtered_renamed = (
    a018_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a018_demo_filtered.columns})
    >> mutate(
        across(
            _[_.endswith("_KF"), _.endswith("_RTS")],
            if_else(Fx < 0, 0, Fx),
        ),
    )
    # >> mutate(
    #     across(
    #         _[_.endswith("_KF"), _.endswith("_RTS")],
    #         if_else(Fx > 1, 1, Fx),
    #     ),
    # )
)

outputs.append(a018_demo_filtered_renamed)

# %%
a018_filtered = pd.concat(outputs, ignore_index=True)

# %% upload
# fdb.upload(
#     df=a018_filtered,
#     database="FUSEDDATA",
#     schema="DATASCI_LAB",
#     table="BIZCATE_M018_AVGSPEND_BM",
#     if_exists="replace",
# )

# %% test
(
    a018_filtered
    >> filter(
        _.CUT_ID == 1,
        _.BIZCATE_CODE == 417, # 417, 111
    )
).plot(
    x="MONTH_YEAR",
    y=[
        "AVG_SPEND",
        "AVG_SPEND_NO_CORR_KF",
        "AVG_SPEND_NO_CORR_RTS",
        "AVG_SPEND_CORR_RTS",
    ],
).legend(loc="best")

# %%
# code 424