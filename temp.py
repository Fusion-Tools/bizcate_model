from typing import Any, Dict
from fusion_kf import Callback

class ProgressCallback(Callback):
    """
    Custom callback class that prints the progress of the run as a percentage.
    """

    def __init__(self, total_partitions: int):
        self.total_partitions = total_partitions
        self.completed_partitions = 0

    def on_model_partition_end(self, models, partition):
        """
        Called when a model partition ends. Prints the progress of the run as a percentage.

        Args:
            models: List of models.
            partition: Current partition.

        Returns:
            pd.DataFrame: The unmodified partition.
        """
        self.completed_partitions += 1
        progress = (self.completed_partitions / self.total_partitions) * 100
        print(f"Progress: {progress:.2f}%")
        return partition


callbacks = [ProgressCallback(total_partitions=100)]

# Run your models and dataloaders with the runner instance
