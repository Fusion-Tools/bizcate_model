# %%
from siuba import *
from db import fdb

# %%
db = "L2SURVEY"
schema = "BIZCATE_ROLLUP"
tbl_name = "BIZCATE_FILTERED_NORMALIZED_M019_AVGSPEND_ECOM"

group_cols = [
    # _.CHANNEL, 
    # _.IMPUTED, 
    _.CUT_ID, 
    # _.OPTION,
    # _.METRIC,
    # _.RETAILER_CODE, 
    _.BIZCATE_CODE,
]

comparison_cols = ("AVG_SPEND", "AVG_SPEND_CORR_RTS")

similarity_threshold = 0.10

# %%

tbl = fdb[db][schema][tbl_name]()

# %%

simlarity_check_data = (
    tbl
    >> filter(_.CUT_ID == 1, _.ASK_COUNT > 0)
    >> group_by(*group_cols)
    >> summarize(
        **{
            f"MEAN_{col}": (getattr(_, col) * _.ASK_COUNT).sum() / _.ASK_COUNT.sum()
            for col in comparison_cols
        }
    )
    >> ungroup()
    >> mutate(
        COMPARISON_DELTA=(
            getattr(_, f"MEAN_{comparison_cols[0]}")
            - getattr(_, f"MEAN_{comparison_cols[1]}")
        ).abs()
    )
    >> mutate(
        COMPARISION_DELTA_PERCENT = _.COMPARISON_DELTA /  getattr(_, f"MEAN_{comparison_cols[0]}")
    )
    >> arrange(-_.COMPARISION_DELTA_PERCENT)
)

# %%

similarity_check_fail_rows = (
    simlarity_check_data
    >> filter(
        _.COMPARISION_DELTA_PERCENT > similarity_threshold
    )
)

# %%
total_rows = simlarity_check_data.shape[0]
fail_rows = similarity_check_fail_rows.shape[0]

print(f"{fail_rows:.0f} of {total_rows} rows ({100 * fail_rows / total_rows : .2f} % ) fails simlarity check for a threshold of {similarity_threshold}.")

# %%
(
    tbl
    >> filter(
        _.CUT_ID == 1,
        # _.METRIC == "TA",
        # _.RETAILER_CODE == 75,
        _.BIZCATE_CODE == 485
    )
).plot(
    x = "MONTH_YEAR",
    y = [
        comparison_cols[0],
        comparison_cols[1],
    ]
)
# %%
