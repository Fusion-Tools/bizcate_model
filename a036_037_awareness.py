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
    10010, 10020, 10030, 10040, 10050, 10060, 10070, 10080, 10090
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


# %% Collect bm and ecom think

bm_think = fetch_raw_think_tom(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A036_BYRETAILER_BMAWARENESS",
    retailers=None,
    channel="BM",
    logit_transform=False,
    cut_ids=CUT_IDS,
)
ecom_think = fetch_raw_think_tom(
    db="L2SURVEY",
    schema="MASPL_ROLLUP",
    tbl_name="A037_BYRETAILER_ECOMAWARENESS",
    retailers=None,
    channel="Ecom",
    logit_transform=False,
    cut_ids=CUT_IDS,
)

think_all = pd.concat([bm_think, ecom_think])

# %%
think_dl = DataLoader(
    table=think_all,
    id_cols=[
        "CUT_ID",
        "CHANNEL",
        "RETAILER_CODE",
        # "SUB_CODE"
    ],
    date_col="MONTH_YEAR",
    var_cols=["SUB_CODE"],
)

tom_kf_module_no_corr = DummyCorrelationKFModule(
    metric_col="TOM",
    sample_size_col="ASK_COUNT",
    process_std=0.03,
    var_corr=0.5,
)

totalthink_kf_module_no_corr = DummyCorrelationKFModule(
    metric_col="TOTALTHINK",
    sample_size_col="ASK_COUNT",
    process_std=0.03,
    var_corr=0.5,
)

# %%

runner = Runner(callbacks=[LogitTransform(), PivotLong(), ConcactPartitions()])

filtered_think = runner.run(
    models=[tom_kf_module_no_corr, totalthink_kf_module_no_corr],
    dataloaders=think_dl,
)

# %%
# fdb.upload(
#     df=filtered_think,
#     database="FUSEDDATA",
#     schema="LEVER_BRAND",
#     table="FILTERED_THINK_MASPL"
# )
# %%
