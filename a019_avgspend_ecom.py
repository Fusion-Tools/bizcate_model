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
def load_avgspend_ecom(
    database="FUSEDDATA",
    schema="LEVER_JSTEP",
    table="BIZCATE_NORMALIZED_AVGSPEND_ECOM",
    cut_ids=None,
):
    a019_avgspend_ecom = (
        fdb[database][schema][table](lazy=True)
        # TODO: temp fix
        >> rename(
            MONTH_NUM=_.MONTH,
            BIZCATE_CODE=_.SUB_CODE,
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

    return a019_avgspend_ecom


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
        Scaler(),
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

outputs = []

# %%
a019_avgspend_ecom = load_avgspend_ecom()


# %%
# ----------------------------------------------------------------
# filter national
# ----------------------------------------------------------------

a019_national = (
    a019_avgspend_ecom
    >> filter(_.CUT_ID == 1)
    >> rename(AVG_SPEND_NATIONAL=_.AVG_SPEND)
)

a019_national_dl = dataloader(a019_national)
a019_national_kf_module_no_corr = no_corr_kf_module("AVG_SPEND_NATIONAL")
a019_national_kf_module_corr = corr_kf_module("AVG_SPEND_NATIONAL")


# %%
a019_national_filtered = runner.run(
    models=[
        a019_national_kf_module_no_corr,
        a019_national_kf_module_corr,
    ],
    dataloaders=a019_national_dl,
    parallel=True
)

# %%
a019_national_filtered_renamed = (
    a019_national_filtered
    >> rename(
        **{col.replace("_NATIONAL", ""): col for col in a019_national_filtered.columns}
    )
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

outputs.append(a019_national_filtered_renamed)

# %%
# ----------------------------------------------------------------
# filter demo cuts using delta method
# ----------------------------------------------------------------

a019_demo = (
    a019_avgspend_ecom
    >> filter(_.CUT_ID != 1)
    >> rename(AVG_SPEND_DEMO=_.AVG_SPEND)
)

# %% calculate demo cuts deltas
a019_demo_delta = (
    inner_join(
        a019_national
        >> select(
            ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT
        ),  # fmt: skip
        a019_demo,
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
a019_demo_delta_dl = dataloader(a019_demo_delta)
a019_demo_kf_module_no_corr = no_corr_kf_module("AVG_SPEND_DELTA")
a019_demo_kf_module_corr = corr_kf_module("AVG_SPEND_DELTA")

a019_demo_delta_filtered = runner.run(
    models=[a019_demo_kf_module_no_corr, a019_demo_kf_module_corr],
    dataloaders=a019_demo_delta_dl,
    parallel=False
)

# %% apply national
a019_demo_filtered = (
    inner_join(
        a019_national_filtered >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT),
        a019_demo_delta_filtered,
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
a019_demo_filtered_renamed = (
    a019_demo_filtered
    >> rename(**{col.replace("_DEMO", ""): col for col in a019_demo_filtered.columns})
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

outputs.append(a019_demo_filtered_renamed)

# %%
a019_filtered = pd.concat(outputs, ignore_index=True)

# %% upload
fdb.upload(
    df=a019_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="BIZCATE_NORMALIZED_M019_AVGSPEND_ECOM",
    if_exists="replace",
)

# %% test
# (
#     a019_filtered
#     >> filter(
#         _.CUT_ID == 1,
#         _.BIZCATE_CODE == 222,
#     )
# ).plot(
#     x="MONTH_YEAR",
#     y=[
#         "AVG_SPEND",
#         "AVG_SPEND_NO_CORR_KF",
#         "AVG_SPEND_NO_CORR_RTS",
#         "AVG_SPEND_CORR_RTS",
#     ],
# ).legend(loc="best")

# %%