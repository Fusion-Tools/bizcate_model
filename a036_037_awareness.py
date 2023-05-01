# %%
from db import fdb
from siuba import *
from siuba.dply.vector import *
import pandas as pd
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule

# %% Define input parameters

# fmt: off
# Define cuts to process (in addition to National)
CUT_IDS = [
    2, 3,
]
# fmt: on

# %% Define prerequisite functions


def complete_table(
    df, identifying_cols, date_col, val_cols, fill_val=None, fill_val_cols=None
):
    # Take the distinct entries from the identifying columns
    unique_df = (
        df[identifying_cols]
        .copy()
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(by=identifying_cols)
        .assign(key=0)
    )

    # Take the distinct entries from the date column
    unique_dates = (
        df.loc[:, [date_col]]
        .copy()
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(by=date_col)
        .assign(key=0)
    )

    # Join the unique identifying columns and dates to create complete table
    expanded_df = unique_df.merge(unique_dates, how="outer")[
        identifying_cols + [date_col]
    ]

    # Join the original dataframe back to the expanded dataframe
    expanded_df = expanded_df.merge(df, how="left", on=identifying_cols + [date_col])

    # If specified, fill the value columns with fill_val
    if fill_val is not None:
        if fill_val_cols is not None:
            for col in fill_val_cols:
                expanded_df[col] = expanded_df[col].fillna(fill_val)

        else:
            for col in val_cols:
                expanded_df[col] = expanded_df[col].fillna(fill_val)

    # Return the completed dataframe
    return expanded_df


def fetch_raw_think_tom(
    db, schema, tbl_name, retailers, channel, logit_transform, cut_ids=None
):
    # Define the month mapping, selecting only required columns
    month_mapping = fdb.LOOKUP.ZZINFO.F005_MONTH(lazy=True) >> select(
        "MONTH_NUM", "MONTH_YEAR"
    )

    # Define the distinct think counts and join month mapping, and collect
    think_counts = (
        fdb[db][schema][tbl_name](lazy=True)
        >> filter(
            _.TOTALTHINK.notna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
            _.RETAILER_CODE.isin(retailers) if retailers is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> distinct(_.CUT_ID, _.SUB_CODE, _.MONTH_YEAR, _.ASK_COUNT, _.ASK_WEIGHT)
        >> rename(FILL_ASK_COUNT="ASK_COUNT", FILL_ASK_WEIGHT="ASK_WEIGHT")
        >> collect()
    )

    # Define the think table, filter, join month mapping, and collect
    think = (
        fdb[db][schema][tbl_name](lazy=True)
        >> filter(
            _.TOTALTHINK.notna(),
            _.CUT_ID.isin([1] + cut_ids) if cut_ids is not None else True,
            _.RETAILER_CODE.isin(retailers) if retailers is not None else True,
        )
        >> left_join(_, month_mapping, on="MONTH_NUM")
        >> select(
            _.CUT_ID,
            _.SUB_CODE,
            _.RETAILER_CODE,
            _.MONTH_YEAR,
            _.ASK_COUNT,
            _.ASK_WEIGHT,
            _.TOM,
            _.TOTALTHINK,
        )
        >> collect()
    )

    # Ensure that the table is complete (set missing entries to 0 ask count and 0 share)
    completed_think = complete_table(
        df=think,
        identifying_cols=["CUT_ID", "SUB_CODE", "RETAILER_CODE"],
        date_col="MONTH_YEAR",
        val_cols=["ASK_COUNT", "ASK_WEIGHT", "TOM", "TOTALTHINK"],
        fill_val_cols=["TOM", "TOTALTHINK"],
        fill_val=0,
    )

    # Join the fill ask counts and weights, and set any remaining missing counts and weights to 0
    completed_think = (
        completed_think
        >> left_join(_, think_counts, on=["CUT_ID", "SUB_CODE", "MONTH_YEAR"])
        >> mutate(
            ASK_COUNT=case_when(
                {
                    _.ASK_COUNT.notna(): _.ASK_COUNT,
                    _.ASK_COUNT.isna() & _.FILL_ASK_COUNT.notna(): _.FILL_ASK_COUNT,
                    True: 0,
                }
            )
        )
        >> mutate(
            ASK_WEIGHT=case_when(
                {
                    _.ASK_WEIGHT.notna(): _.ASK_WEIGHT,
                    _.ASK_WEIGHT.isna() & _.FILL_ASK_WEIGHT.notna(): _.FILL_ASK_WEIGHT,
                    True: 0,
                }
            )
        )
    )

    # # Set the channel column and re-order
    completed_think["CHANNEL"] = channel
    completed_think = completed_think[
        [
            "CHANNEL",
            "RETAILER_CODE",
            "SUB_CODE",
            "MONTH_YEAR",
            "ASK_COUNT",
            "ASK_WEIGHT",
            "TOM",
            "TOTALTHINK",
            "CUT_ID",
        ]
    ]
    completed_think = completed_think.sort_values(
        by=["CUT_ID", "RETAILER_CODE", "SUB_CODE", "MONTH_YEAR"]
    )

    # Convert to logit space if specified in the function call
    if logit_transform:
        completed_think["TOM"] = logit(completed_think["TOM"])
        completed_think["TOTALTHINK"] = logit(completed_think["TOTALTHINK"])

    return completed_think


# %% Collect bm and ecom think for maspl

bm_think_maspl = fetch_raw_think_tom(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A036_BYRETAILER_BMAWARENESS",
    retailers=None,
    channel="BM",
    logit_transform=False,
    cut_ids=CUT_IDS,
)
ecom_think_maspl = fetch_raw_think_tom(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A037_BYRETAILER_ECOMAWARENESS",
    retailers=None,
    channel="Ecom",
    logit_transform=False,
    cut_ids=CUT_IDS,
)

think_maspl = pd.concat([bm_think_maspl, ecom_think_maspl])

# %% 
# Define the Bizcate mapping table
bizcate_sub_mapping = (
    fdb.FUSEDDATA.LEVER_JSTEP.LOOKUP_BIZCATE_SUBCATE_QUOTA(lazy=True)
    >> filter(_.BIZCATE_CODE.notna())
    >> distinct(_.BIZCATE_CODE, _.SUB_CODE)
    >> collect()
)

# %% Collect bm and ecom think for bizcate
bm_think_bizcate = fetch_raw_think_tom(
    db="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    tbl_name="A036_BYRETAILER_BMAWARENESS",
    retailers=None,
    channel="BM",
    logit_transform=False,
    cut_ids=CUT_IDS,
)
ecom_think_bizcate = fetch_raw_think_tom(
    db="L2SURVEY",
    schema="BIZCATE_ROLLUP",
    tbl_name="A037_BYRETAILER_ECOMAWARENESS",
    retailers=None,
    channel="Ecom",
    logit_transform=False,
    cut_ids=CUT_IDS,
)

think_bizcate = (
    pd.concat([bm_think_bizcate, ecom_think_bizcate])
    >> rename(BIZCATE_CODE=_.SUB_CODE)
    >> left_join(_, bizcate_sub_mapping, on="BIZCATE_CODE")
)


# %%
def fetch_filtered_think_tom(
    db="FUSEDDATA",
    schema="LEVER_BRAND",
    tbl_name="COMPSHARE_011_FILTERED_SURVEY_THINK",
):
    f004_geo_reported = (
        fdb.LOOKUP.ZZINFO.F004_GEO_REPORTED(lazy=True)
        >> filter(_.REPORTED_REGION_CODE.isin([113, 116]))
        >> select(_.FUSION_REGION_CODE, _.REPORTED_REGION_CODE)
    )

    audited_001_industry = (
        fdb.L2METRICS.DASH_ZZ_COMMON.AUDITED_001_INDUSTRYSIZE(lazy=True)
        >> filter(_.CUSTOMER_TYPE == "Consumer Retail", _.CHANNEL.isin(["BM", "Ecom"]))
        >> distinct(
            _.CHANNEL,
            _.FUSION_REGION_CODE,
            _.SUB_CODE,
            _.MONTH_YEAR,
            _.INDUSTRY_SIZE,
        )
    )

    think_national = (
        fdb[db][schema][tbl_name](lazy=True)
        >> select(
            _.FUSION_REGION_CODE,
            _.CHANNEL,
            _.RETAILER_CODE,
            _.SUB_CODE,
            _.MONTH_YEAR,
            _.TOM_SMOOTHED,
            _.TOTALTHINK_SMOOTHED,
        )
        >> filter(_.FUSION_REGION_CODE == 0)
        >> mutate(CUT_ID=case_when(_, {_.FUSION_REGION_CODE == 0: 1, True: 0}))
        >> select(~_.FUSION_REGION_CODE)
        >> collect()
        >> mutate(MONTH_YEAR=_.MONTH_YEAR.dt.date)
    )

    think_regions = (
        fdb[db][schema][tbl_name](lazy=True)
        >> select(
            _.FUSION_REGION_CODE,
            _.CHANNEL,
            _.RETAILER_CODE,
            _.SUB_CODE,
            _.MONTH_YEAR,
            _.TOM_SMOOTHED,
            _.TOTALTHINK_SMOOTHED,
        )
        >> inner_join(
            _,
            audited_001_industry,
            on=["CHANNEL", "FUSION_REGION_CODE", "SUB_CODE", "MONTH_YEAR"],
        )
        >> inner_join(
            _,
            f004_geo_reported,
            on=["FUSION_REGION_CODE"],
        )
        >> mutate(
            WEIGHTED_TOM_SMOOTHED=_.TOM_SMOOTHED * _.INDUSTRY_SIZE,
            WEIGHTED_TOTALTHINK_SMOOTHED=_.TOTALTHINK_SMOOTHED * _.INDUSTRY_SIZE,
        )
        >> group_by(
            _.REPORTED_REGION_CODE,
            _.CHANNEL,
            _.RETAILER_CODE,
            _.SUB_CODE,
            _.MONTH_YEAR,
        )
        >> summarize(
            SUM_WEIGHTED_TOM_SMOOTHED=_.WEIGHTED_TOM_SMOOTHED.sum(),
            SUM_WEIGHTED_TOTALTHINK_SMOOTHED=_.WEIGHTED_TOTALTHINK_SMOOTHED.sum(),
            SUM_INDUSTRY_SIZE=_.INDUSTRY_SIZE.sum(),
        )
        >> ungroup()
        >> mutate(
            TOM_SMOOTHED=_.SUM_WEIGHTED_TOM_SMOOTHED / _.SUM_INDUSTRY_SIZE,
            TOTALTHINK_SMOOTHED=_.SUM_WEIGHTED_TOTALTHINK_SMOOTHED
            / _.SUM_INDUSTRY_SIZE,
        )
        >> mutate(
            CUT_ID=case_when(
                _,
                {
                    _.REPORTED_REGION_CODE == 116: 2,
                    _.REPORTED_REGION_CODE == 113: 3,
                    True: 0,
                },
            )
        )
        >> select(
            ~_.SUM_WEIGHTED_TOM_SMOOTHED,
            ~_.SUM_WEIGHTED_TOTALTHINK_SMOOTHED,
            ~_.SUM_INDUSTRY_SIZE,
            ~_.REPORTED_REGION_CODE,
        )
        >> collect()
        >> mutate(MONTH_YEAR=_.MONTH_YEAR.dt.date)
    )

    think_filtered = pd.concat([think_national, think_regions])

    return think_filtered


# %%
think_maspl_filtered = fetch_filtered_think_tom(
    db="FUSEDDATA",
    schema="LEVER_BRAND",
    tbl_name="COMPSHARE_011_FILTERED_SURVEY_THINK",
)


# %%
def dataloader(table):
    return DataLoader(
        table=table,
        id_cols=[
            "CUT_ID",
            "CHANNEL",
            "RETAILER_CODE",
        ],
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
        LogitTransform(),
        PivotLong(),
        ConcactPartitions()
    ]
)
# fmt: on

# %%
think_maspl_national = (
    think_maspl
    >> filter(_.CUT_ID == 1)
    >> rename(TOM_MASPL=_.TOM, TOTALTHINK_MASPL=_.TOTALTHINK)
)
think_bizcate_national = think_bizcate >> filter(_.CUT_ID == 1)

# %%
# %% calculate national deltas to maspl
think_bizcate_national_delta = (
    inner_join(
        think_maspl_national >> select(~_.ASK_COUNT, ~_.ASK_WEIGHT),
        think_bizcate_national,
        on=[
            "CUT_ID",
            "CHANNEL",
            "RETAILER_CODE",
            "SUB_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        TOM_DELTA=_.TOM_MASPL - _.TOM,  # fmt: skip
        TOTALTHINK_DELTA=_.TOTALTHINK_MASPL - _.TOTALTHINK  # fmt: skip
    )
    >> select(~_.SUB_CODE, ~_.TOM_MASPL, ~_.TOTALTHINK_MASPL)
)

# %% filter bizcate deltas to national
think_bizcate_national_delta_dl = dataloader(think_bizcate_national_delta)
tom_kf_module_no_corr = no_corr_kf_module("TOM_DELTA")
tom_kf_module_corr = corr_kf_module("TOM_DELTA")
think_kf_module_no_corr = no_corr_kf_module("TOTALTHINK_DELTA")
think_kf_module_corr = corr_kf_module("TOTALTHINK_DELTA")

think_bizcate_national_delta_filtered = runner.run(
    models=[
        tom_kf_module_no_corr,
        tom_kf_module_corr,
        think_kf_module_no_corr,
        think_kf_module_corr,
    ],
    dataloaders=think_bizcate_national_delta_dl,
)

# %% apply filtered maspl
think_maspl_national_filtered = (
    think_maspl_filtered 
    >> filter(_.CUT_ID == 1)
    >> inner_join(_, bizcate_sub_mapping, on="SUB_CODE")
)

think_bizcate_national_filtered = (
    inner_join(
        think_maspl_national_filtered,
        think_bizcate_national_delta_filtered,
        on=[
            "CUT_ID",
            "CHANNEL",
            "RETAILER_CODE",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> select(
        ~_.endswith("KF")
    )
    # fmt: off
    >> mutate(
        TOM_NO_CORR_RTS=_.TOM_SMOOTHED - _.TOM_DELTA_NO_CORR_RTS,
        TOTALTHINK_NO_CORR_RTS=_.TOTALTHINK_SMOOTHED - _.TOTALTHINK_DELTA_NO_CORR_RTS,
        TOM_CORR_RTS=_.TOM_SMOOTHED - _.TOM_DELTA_CORR_RTS,
        TOTALTHINK_CORR_RTS=_.TOTALTHINK_SMOOTHED - _.TOTALTHINK_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.SUB_CODE,
        ~_.TOM_DELTA,
        ~_.TOM_DELTA_NO_CORR_RTS,
        ~_.TOM_DELTA_CORR_RTS,
        ~_.TOTALTHINK_DELTA,
        ~_.TOTALTHINK_DELTA_NO_CORR_RTS,
        ~_.TOTALTHINK_DELTA_CORR_RTS,
        ~_.TOM_SMOOTHED,
        ~_.TOTALTHINK_SMOOTHED,
    )
)

# %% calculate regions deltas to bizcate natioanl
think_bizcate_regional = (
    think_bizcate 
    >> filter(_.CUT_ID != 1)
   
)

# %% 
think_bizcate_regional_delta = (
    inner_join(
        (   
            think_bizcate_national 
            >> select(~_.SUB_CODE, ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT)
            >> rename(TOM_NATIONAL=_.TOM, TOTALTHINK_NATIONAL=_.TOTALTHINK)
        ),
        think_bizcate_regional >> select(~_.SUB_CODE),
        on=[
            "CHANNEL",
            "RETAILER_CODE",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        TOM_DELTA=_.TOM_NATIONAL - _.TOM,  # fmt: skip
        TOTALTHINK_DELTA=_.TOTALTHINK_NATIONAL - _.TOTALTHINK  # fmt: skip
    )
    >> select( ~_.TOM_NATIONAL, ~_.TOTALTHINK_NATIONAL)
)

# %% filter demo cuts deltas to national
think_bizcate_regional_delta_dl = dataloader(think_bizcate_regional_delta)

think_bizcate_regional_delta_filtered = runner.run(
    models=[
        tom_kf_module_no_corr,
        tom_kf_module_corr,
        think_kf_module_no_corr,
        think_kf_module_corr,
    ],
    dataloaders=think_bizcate_regional_delta_dl,
)

# %% apply filtered national

think_bizcate_regional_filtered = (
    inner_join(
        (
            think_bizcate_national_filtered 
            >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT, ~_.TOM, ~_.TOTALTHINK)
            >> rename(
                TOM_NO_CORR_RTS_NATIONAL = _.TOM_NO_CORR_RTS,	
                TOTALTHINK_NO_CORR_RTS_NATIONAL = _.TOTALTHINK_NO_CORR_RTS,	
                TOM_CORR_RTS_NATIONAL = _.TOM_CORR_RTS,	
                TOTALTHINK_CORR_RTS_NATIONAL = _.TOTALTHINK_CORR_RTS,
            )
        ),
        think_bizcate_regional_delta_filtered,
        on=[
            "CHANNEL",
            "RETAILER_CODE",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> select(
        ~_.endswith("KF")
    )
    # fmt: off
    >> mutate(
        TOM_NO_CORR_RTS=_.TOM_NO_CORR_RTS_NATIONAL - _.TOM_DELTA_NO_CORR_RTS,
        TOTALTHINK_NO_CORR_RTS=_.TOTALTHINK_NO_CORR_RTS_NATIONAL - _.TOTALTHINK_DELTA_NO_CORR_RTS,
        TOM_CORR_RTS=_.TOM_CORR_RTS_NATIONAL - _.TOM_DELTA_CORR_RTS,
        TOTALTHINK_CORR_RTS=_.TOTALTHINK_CORR_RTS_NATIONAL - _.TOTALTHINK_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.TOM_DELTA,
        ~_.TOM_DELTA_NO_CORR_RTS,
        ~_.TOM_DELTA_CORR_RTS,
        ~_.TOTALTHINK_DELTA,
        ~_.TOTALTHINK_DELTA_NO_CORR_RTS,
        ~_.TOTALTHINK_DELTA_CORR_RTS,
        ~_.TOM_NO_CORR_RTS_NATIONAL,
        ~_.TOTALTHINK_NO_CORR_RTS_NATIONAL,
        ~_.TOM_CORR_RTS_NATIONAL,
        ~_.TOTALTHINK_CORR_RTS_NATIONAL
    )
)

# %%
think_bizcate_filtered = (
    pd.concat(
        [think_bizcate_national_filtered, think_bizcate_regional_filtered]
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

# %%
# fdb.upload(
#     df=think_bizcate_filtered,
#     database="FUSEDDATA",
#     schema="DATASCI_LAB",
#     table="BIZCATE_M036_037_AWARENESS"
# )

# %%
# (
#     think_bizcate_filtered
#     >> filter(
#         _.CHANNEL == "BM",
#         _.CUT_ID == 1,
#         _.BIZCATE_CODE == 102,
#         _.RETAILER_CODE == 115,
#     )
# ).plot(
#     x = "MONTH_YEAR",
#     y = [
#         "TOM",
#         "TOM_NO_CORR_RTS",
#         "TOM_CORR_RTS",
#     ]
# ).legend(loc='best')
# %%
