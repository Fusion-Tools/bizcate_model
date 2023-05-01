# %%
import os
from db import fdb
from siuba import *
from siuba.dply.vector import *
import pyarrow as pa
import pandas as pd
import datetime as dt


# %% Read in the Think data


def fetch_filtered_bizcate():
    # Define the filtered Bizcate Think data
    bizcate_think = (
        fdb.FUSEDDATA.LEVER_BRAND.FILTERED_THINK_BIZCATE(lazy=True)
        >> filter(
            _.CUT_ID == 1,
            _.RETAILER_CODE.notna(),
            _.SUB_CODE.notna(),
        )
        >> collect()
    )

    return bizcate_think


def calculate_category_correlation(
    survey_df,
    identifying_cols,
    category_col,
    value_col,
    min_periods,
    ensure_positive_definite=True,
):
    # Calculate the correlation matrix
    survey_correlation = (
        survey_df.pivot(index=identifying_cols, columns=category_col, values=value_col)
        .corr(method="pearson", min_periods=min_periods)
        .fillna(0)
    )

    # Fill the main diagonal with 1s (to ensure low sample subcategories are correct)
    survey_correlation.values[np.diag_indices_from(survey_correlation.values)] = 1

    # If specified, correct the correlation matrix to ensure that it's positive definite
    if ensure_positive_definite:
        corrected_survey_correlation = correct_correlation_matrix(survey_correlation)
        survey_correlation = pd.DataFrame(
            corrected_survey_correlation,
            index=survey_correlation.index,
            columns=survey_correlation.columns,
        )

    # Square the correlation matrix
    survey_correlation = np.square(survey_correlation)

    # Prepare the long-form correlation table
    survey_correlation_long = (
        pd.melt(survey_correlation, value_name="CORRELATION", ignore_index=False)
        .rename(columns={"SUB_CODE": "SIMILAR_SUB_CODE"})
        .sort_values(["SUB_CODE", "CORRELATION"], ascending=[True, False])
        .reset_index()
    )

    return survey_correlation_long


def correct_correlation_matrix(cor_matrix, min_value=1e-5):
    """Given initial correlation matrix, ensure that the result is positive (semi)definite"""

    # Get the eigenvalues and eigenvectors and convert to matrices
    eigval, eigvec = np.linalg.eig(cor_matrix)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, min_value)))

    # Calculate the corrected correlation matrix
    corrected_cor_matrix = Q * xdiag * Q.T

    # Scale the corrected correlation matrix
    correction_factor = np.matrix(np.diag(1 / np.sqrt(np.diag(corrected_cor_matrix))))

    # Apply the correction factor
    corrected_cor_matrix = (
        correction_factor * corrected_cor_matrix * correction_factor.T
    )

    return corrected_cor_matrix


def fetch_bizcate_mapping():
    # Define the Bizcate mapping table
    bizcate_mapping = (
        fdb.FUSEDDATA.LEVER_JSTEP.LOOKUP_BIZCATE_SUBCATE_QUOTA(lazy=True)
        >> filter(
            _.BIZCATE_CODE.notna(),
        )
        >> distinct(_.BIZCATE_CODE, _.BIZCATE)
        >> collect()
    )

    return bizcate_mapping


# %% Pull Bizcategory Think data
bizcate_think = fetch_filtered_bizcate()

# %% Calculate correlation matrix
bizcate_correlation = calculate_category_correlation(
    survey_df=bizcate_think,
    identifying_cols=["CHANNEL", "CUT_ID", "MONTH_YEAR", "RETAILER_CODE"],
    category_col="SUB_CODE",
    value_col="TOTALTHINK_RTS",
    min_periods=3500,
).rename(
    columns={"SUB_CODE": "BIZCATE_CODE", "SIMILAR_SUB_CODE": "SIMILAR_BIZCATE_CODE"}
)


# %% Upload the correlation matrix
# fdb.upload(
#     df=bizcate_correlation,
#     database="FUSEDDATA",
#     schema="LEVER_BRAND",
#     table="BIZCATE_THINK_CORRELATION",
#     if_exists="replace",
# )

# %% Check the results of the correlation

# Get the Bizcate mapping
bizcate_mapping = fetch_bizcate_mapping()

# Join the bizcate mapping to the main table
formatted_bizcate_correlation = bizcate_correlation.merge(
    bizcate_mapping, how="left", on="BIZCATE_CODE"
).merge(
    bizcate_mapping.rename(
        columns={"BIZCATE": "SIMILAR_BIZCATE", "BIZCATE_CODE": "SIMILAR_BIZCATE_CODE"}
    ),
    how="left",
    on="SIMILAR_BIZCATE_CODE",
) >> select(
    _.BIZCATE_CODE, _.BIZCATE, _.SIMILAR_BIZCATE_CODE, _.SIMILAR_BIZCATE, _.CORRELATION
)

# %%
(
    formatted_bizcate_correlation >> filter(_.BIZCATE_CODE == 117)
).CORRELATION.to_numpy()

# %%