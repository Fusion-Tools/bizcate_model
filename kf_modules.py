# %%
# %%
import os

# os.environ["PANDAS_COPY_ON_WRITE"] = "1"

# %%
import pandas as pd

pd.set_option("display.memory_usage", "deep")

# %%
from fusion_kf import KFModule
from siuba import *
import numpy as np
from db import fdb
import pyarrow as pa

# %%


def full_bizcat_corr_matrix():
    corr_bizcats_df = (
        fdb.FUSEDDATA.LEVER_BRAND.BIZCATE_THINK_CORRELATION(lazy=True)
        >> select(_.BIZCATE_CODE, _.SIMILAR_BIZCATE_CODE, _.CORRELATION)
        >> arrange(_.BIZCATE_CODE, _.SIMILAR_BIZCATE_CODE)
        >> collect()
    ).set_index(["BIZCATE_CODE", "SIMILAR_BIZCATE_CODE"])

    bizcats_with_corr = corr_bizcats_df.index.get_level_values("BIZCATE_CODE").unique()
    complete_bizcat_index = pd.RangeIndex(
        min(bizcats_with_corr), max(bizcats_with_corr) + 1
    )
    multi_index = pd.MultiIndex.from_product(
        [complete_bizcat_index, complete_bizcat_index],
        names=["BIZCATE_CODE", "SIMILAR_BIZCATE_CODE"],
    )

    corr_all_bizcats_df = (
        corr_bizcats_df.reindex(multi_index, fill_value=0)  # FIXME: temp imputation
        .reset_index()
        .pivot_table(index="BIZCATE_CODE", columns="SIMILAR_BIZCATE_CODE", values="CORRELATION")
    )

    return corr_all_bizcats_df


# %%
class BizcateCorrelationKFModule(KFModule):
    def __init__(
        self, *, metric_col, output_col_prefix=None, sample_size_col, process_std
    ):
        self.sample_size_col = sample_size_col
        self.process_std = process_std
        self.corr_all_bizcats_df = full_bizcat_corr_matrix()
        super().__init__(metric_col=metric_col, output_col_prefix=output_col_prefix)

    def process_covariance(self, raw_df):
        bizcats = raw_df.columns.get_level_values("BIZCATE_CODE").unique().to_numpy()
        corr_bizcats = self.corr_all_bizcats_df.loc[bizcats, bizcats].to_numpy()
        # Q_bizcats (across bizcats)
        cov_bizcats = corr_bizcats * (self.process_std**2)
        return cov_bizcats

    def measurement_covariance(self, raw_df):
        zs = raw_df.loc[:, [self.metric_col]].to_numpy(copy=True)
        zs[np.isnan(zs)] = 0

        ns = raw_df.loc[:, [self.sample_size_col]].to_numpy(copy=True)
        ns[np.isnan(ns)] = 0

        se = np.sqrt(np.abs(zs * (1 - zs) + 1e-8) / (ns + 1e-8)) + np.sqrt(0.25 / (ns + 1e-8))  # fmt: skip
        se_2 = se**2
        Rs = np.eye(se_2.shape[1]) * se_2[:, np.newaxis, :]
        return Rs
