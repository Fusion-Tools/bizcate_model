# %%
import os
from db import fdb
from siuba import *
from siuba.dply.vector import *
from fusion_kf import DataLoader, Runner
from fusion_kf.kf_modules import NoCorrelationKFModule
from fusion_kf.callbacks import LogitTransform, PivotLong, ConcactPartitions
from kf_modules import BizcateCorrelationKFModule

# %% Define input parameters

# fmt: off
# Define cuts to process (in addition to National)
CUT_IDS = [
    # 2, 3,
    10010, 10020, 10030, 10040, 10050, 10060, 10070, 10080, 10090, # Regions
]
# fmt: on


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


def fetch_raw_market_share(
    db, schema, tbl_name, retailers, channel, logit_transform, cut_ids=None
):
    # Define the month mapping, selecting only required columns
    month_mapping = fdb.LOOKUP.ZZINFO.F005_MONTH(lazy=True) >> select(
        "MONTH_NUM", "MONTH_YEAR"
    )

    # Define the market share table, filter, join month mapping, and collect
    market_share = (
        fdb[db][schema][tbl_name](lazy=True)
        >> filter(
            _.SHARE_FINAL.notna(),
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
            _.SHARE_FINAL,
        )
        >> collect()
    )

    # Ensure that the table is complete (set missing entries to 0 ask count and 0 share)
    completed_market_share = complete_table(
        df=market_share,
        identifying_cols=["CUT_ID", "SUB_CODE", "RETAILER_CODE"],
        date_col="MONTH_YEAR",
        val_cols=["ASK_COUNT", "ASK_WEIGHT", "SHARE_FINAL"],
        fill_val=0,
    )

    # Set the channel column and re-order
    completed_market_share["CHANNEL"] = channel
    completed_market_share = completed_market_share[
        [
            "CHANNEL",
            "RETAILER_CODE",
            "SUB_CODE",
            "MONTH_YEAR",
            "ASK_COUNT",
            "ASK_WEIGHT",
            "SHARE_FINAL",
            "CUT_ID",
        ]
    ]
    completed_market_share = completed_market_share.sort_values(
        by=["CUT_ID", "RETAILER_CODE", "SUB_CODE", "MONTH_YEAR"]
    )

    # Convert to logit space if specified in the function call
    if logit_transform:
        completed_market_share["SHARE_FINAL"] = logit(
            completed_market_share["SHARE_FINAL"]
        )

    # Return the dataframe
    return completed_market_share


# %% Collect bm and ecom market share

bm_marketshare_maspl = fetch_raw_market_share(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A043_BYRETAILER_BMBASE_MARKETSHARE",
    retailers=None,
    channel="BM",
    logit_transform=False,
    cut_ids=CUT_IDS,
)
ecom_marketshare_maspl = fetch_raw_market_share(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A046_BYRETAILER_ECOMBASE_MARKETSHARE",
    retailers=None,
    channel="Ecom",
    logit_transform=False,
    cut_ids=CUT_IDS,
)

marketshare_maspl = pd.concat([bm_marketshare_maspl, ecom_marketshare_maspl]) >> rename(
    MARKET_SHARE=_.SHARE_FINAL
)


# %% Define the Bizcate mapping table
bizcate_sub_mapping = (
    fdb.FUSEDDATA.LEVER_JSTEP.LOOKUP_BIZCATE_SUBCATE_QUOTA(lazy=True)
    >> filter(_.BIZCATE_CODE.notna())
    >> distinct(_.BIZCATE_CODE, _.SUB_CODE)
    >> collect()
)

# %% collect bm and ecom market share for bizcate
bm_marketshare_bizcate = fetch_raw_market_share(
    db="FUSEDDATA",
    schema="LEVER_JSTEP",
    tbl_name="BIZCATE_NORMALIZED_BYRETAILER_BMBASE_MARKETSHARE",
    retailers=None,
    channel="BM",
    logit_transform=False,
    cut_ids=CUT_IDS,
)
ecom_marketshare_bizcate = fetch_raw_market_share(
    db="FUSEDDATA",
    schema="LEVER_JSTEP",
    tbl_name="BIZCATE_NORMALIZED_BYRETAILER_ECOMBASE_MARKETSHARE",
    retailers=None,
    channel="Ecom",
    logit_transform=False,
    cut_ids=CUT_IDS,
)

marketshare_bizcate = (
    pd.concat([bm_marketshare_bizcate, ecom_marketshare_bizcate])
    >> rename(BIZCATE_CODE=_.SUB_CODE, MARKET_SHARE=_.SHARE_FINAL)
    >> left_join(_, bizcate_sub_mapping, on="BIZCATE_CODE")
)


# %%
# %%
def fetch_filtered_market_share(
    db="L2METRICS",
    schema="DASH_ZZ_COMMON",
    tbl_name="AUDITED_011_BANNERSHARE",
):
    marketshare_national = (
        fdb[db][schema][tbl_name](lazy=True)
        >> select(
            _.FUSION_REGION_CODE,
            _.CHANNEL,
            _.RETAILER_CODE,
            _.SUB_CODE,
            _.MONTH_YEAR,
            _.MARKET_SHARE,
        )
        >> filter(_.FUSION_REGION_CODE == 0, _.CHANNEL.isin(["BM", "Ecom"]))
        >> mutate(CUT_ID=case_when(_, {_.FUSION_REGION_CODE == 0: 1, True: 0}))
        >> select(~_.FUSION_REGION_CODE)
        >> rename(MARKET_SHARE_SMOOTHED=_.MARKET_SHARE)
        >> collect()
    )

    return marketshare_national


# %%
marketshare_maspl_national_filtered = fetch_filtered_market_share(
    db="L2METRICS",
    schema="DASH_ZZ_COMMON",
    tbl_name="AUDITED_011_BANNERSHARE",
)


# %%
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
marketshare_maspl_national = (
    marketshare_maspl
    >> filter(_.CUT_ID == 1)
    >> rename(MARKET_SHARE_MASPL=_.MARKET_SHARE)
)
marketshare_bizcate_national = marketshare_bizcate >> filter(_.CUT_ID == 1)

# %% calculate national deltas to maspl
marketshare_bizcate_national_delta = (
    inner_join(
        marketshare_maspl_national >> select(~_.ASK_COUNT, ~_.ASK_WEIGHT),
        marketshare_bizcate_national,
        on=[
            "CUT_ID",
            "CHANNEL",
            "RETAILER_CODE",
            "SUB_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        MARKET_SHARE_DELTA=_.MARKET_SHARE_MASPL - _.MARKET_SHARE,  # fmt: skip
    )
    >> select(~_.SUB_CODE, ~_.MARKET_SHARE_MASPL)
)

# %% filter bizcate deltas to national
marketshare_bizcate_national_delta_dl = dataloader(marketshare_bizcate_national_delta)
marketshare_kf_module_no_corr = no_corr_kf_module("MARKET_SHARE_DELTA")
marketshare_kf_module_corr = corr_kf_module("MARKET_SHARE_DELTA")

marketshare_bizcate_national_delta_filtered = runner.run(
    models=[
        marketshare_kf_module_no_corr,
        marketshare_kf_module_corr,
    ],
    dataloaders=marketshare_bizcate_national_delta_dl,
)

# %% apply filtered maspl

marketshare_bizcate_national_filtered = (
    inner_join(
        (
            marketshare_maspl_national_filtered
            >> inner_join(_, bizcate_sub_mapping, on="SUB_CODE")
        ),
        marketshare_bizcate_national_delta_filtered,
        on=[
            "CUT_ID",
            "CHANNEL",
            "RETAILER_CODE",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> select(~_.endswith("KF"))
    # fmt: off
    >> mutate(
        MARKET_SHARE_NO_CORR_RTS=_.MARKET_SHARE_SMOOTHED - _.MARKET_SHARE_DELTA_NO_CORR_RTS,
        MARKET_SHARE_CORR_RTS=_.MARKET_SHARE_SMOOTHED - _.MARKET_SHARE_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.SUB_CODE,
        ~_.MARKET_SHARE_DELTA,
        ~_.MARKET_SHARE_DELTA_NO_CORR_RTS,
        ~_.MARKET_SHARE_DELTA_CORR_RTS,
        ~_.MARKET_SHARE_SMOOTHED,
    )
)

# %% calculate regions deltas to bizcate natioanl
marketshare_bizcate_regional = marketshare_bizcate >> filter(_.CUT_ID != 1)

# %%
marketshare_bizcate_regional_delta = (
    inner_join(
        (
            marketshare_bizcate_national
            >> select(~_.SUB_CODE, ~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT)
            >> rename(MARKET_SHARE_NATIONAL=_.MARKET_SHARE)
        ),
        marketshare_bizcate_regional >> select(~_.SUB_CODE),
        on=[
            "CHANNEL",
            "RETAILER_CODE",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> mutate(
        MARKET_SHARE_DELTA=_.MARKET_SHARE_NATIONAL - _.MARKET_SHARE,  # fmt: skip
    )
    >> select(~_.MARKET_SHARE_NATIONAL)
)

# %% filter demo cuts deltas to national
marketshare_bizcate_regional_delta_dl = dataloader(marketshare_bizcate_regional_delta)

marketshare_bizcate_regional_delta_filtered = runner.run(
    models=[
        marketshare_kf_module_no_corr,
        marketshare_kf_module_corr,
    ],
    dataloaders=marketshare_bizcate_regional_delta_dl,
)

# %% apply filtered national

marketshare_bizcate_regional_filtered = (
    inner_join(
        (
            marketshare_bizcate_national_filtered
            >> select(~_.CUT_ID, ~_.ASK_COUNT, ~_.ASK_WEIGHT, ~_.MARKET_SHARE)
            >> rename(
                MARKET_SHARE_NO_CORR_RTS_NATIONAL=_.MARKET_SHARE_NO_CORR_RTS,
                MARKET_SHARE_CORR_RTS_NATIONAL=_.MARKET_SHARE_CORR_RTS,
            )
        ),
        marketshare_bizcate_regional_delta_filtered,
        on=[
            "CHANNEL",
            "RETAILER_CODE",
            "BIZCATE_CODE",
            "MONTH_YEAR",
        ],
    )
    >> select(~_.endswith("KF"))
    # fmt: off
    >> mutate(
        MARKET_SHARE_NO_CORR_RTS=_.MARKET_SHARE_NO_CORR_RTS_NATIONAL - _.MARKET_SHARE_DELTA_NO_CORR_RTS,
        MARKET_SHARE_CORR_RTS=_.MARKET_SHARE_CORR_RTS_NATIONAL - _.MARKET_SHARE_DELTA_CORR_RTS,
    )
    # fmt:on
    >> select(
        ~_.MARKET_SHARE_DELTA,
        ~_.MARKET_SHARE_DELTA_NO_CORR_RTS,
        ~_.MARKET_SHARE_DELTA_CORR_RTS,
        ~_.MARKET_SHARE_NO_CORR_RTS_NATIONAL,
        ~_.MARKET_SHARE_CORR_RTS_NATIONAL,
    )
)


# %%
# %%
marketshare_bizcate_filtered = (
    pd.concat(
        [marketshare_bizcate_national_filtered, marketshare_bizcate_regional_filtered]
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
fdb.upload(
    df=marketshare_bizcate_filtered,
    database="FUSEDDATA",
    schema="DATASCI_LAB",
    table="BIZCATE_NORMALIZED_M043_046_MARKETSHARE",
)

# %%
# (
#     marketshare_bizcate_filtered
#     >> filter(
#         _.CHANNEL == "BM",
#         _.CUT_ID == 1,
#         _.BIZCATE_CODE == 111,
#         _.RETAILER_CODE == 45,
#     )
# ).plot(
#     x="MONTH_YEAR",
#     y=[
#         "MARKET_SHARE",
#         "MARKET_SHARE_NO_CORR_RTS",
#         "MARKET_SHARE_CORR_RTS",
#     ],
# ).legend(
#     loc="best"
# )


# %%
