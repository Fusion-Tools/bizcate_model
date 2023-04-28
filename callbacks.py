from typing import Any, Dict
from fusion_kf import Callback
from sklearn.preprocessing import StandardScaler


class Scaler(Callback):
    """
    Custom callback class that tarnsforms and inverse_transforms metric cols.
    """

    def __init__(
        self,
    ):
        self._fitted_scalers = dict()

    def on_model_partition_start(self, models, partition):
        levels_to_keep = partition.index.nlevels - 1
        partition_key = list(
            set([tuple(idx[:levels_to_keep]) for idx in partition.index])
        )[0]

        # transforms unique set of metric cols when multiple models are specified
        cols_to_transform = list(set([model.metric_col for model in models]))

        scalers = {}
        # Iterate through the columns and scale each one individually
        for metric_col in cols_to_transform:
            for col in partition[[metric_col]].columns:
                scaler = StandardScaler()
                partition.loc[:, col] = scaler.fit_transform(partition[[col]])
                scalers[col] = scaler
        self._fitted_scalers[partition_key] = scalers
        return partition

    def on_model_partition_end(self, models, partition):
        levels_to_keep = partition.index.nlevels - 1
        partition_key = list(
            set([tuple(idx[:levels_to_keep]) for idx in partition.index])
        )[0]

        scalers = self._fitted_scalers[partition_key]

        # transforms unique set of metric & output cols
        # when multiple models are specified
        col_sets_to_transform = list(
            set([(model.metric_col, *model.output_cols) for model in models])
        )

        transformed = []
        # Use the stored scalers for inverse_transform
        for col_set in col_sets_to_transform:
            for metric_col in col_set:
                if metric_col not in transformed:
                    for col in partition[[metric_col]].columns:
                        scaler = scalers[(col_set[0], *col[1:])]
                        partition.loc[:, col] = scaler.inverse_transform(partition[[col]])  # fmt: skip

                    transformed.append(metric_col)

        return partition
