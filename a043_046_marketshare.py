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

# Define cuts to process (in addition to National)
CUT_IDS = [10010, 10020, 10030, 10040, 10050, 10060, 10070, 10080, 10090]


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

bm_market_share = fetch_raw_market_share(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A043_BYRETAILER_BMBASE_MARKETSHARE",
    retailers=None,
    channel="BM",
    logit_transform=False,
    cut_ids=CUT_IDS,
)
ecom_market_share = fetch_raw_market_share(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A046_BYRETAILER_ECOMBASE_MARKETSHARE",
    retailers=None,
    channel="Ecom",
    logit_transform=False,
    cut_ids=CUT_IDS,
)
