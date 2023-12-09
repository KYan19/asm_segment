import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import NonGeoDataset, stack_samples, unbind_samples
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.trainers import PixelwiseRegressionTask, SemanticSegmentationTask
from torchvision.transforms.functional import pad
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import geopandas as gpd
import rasterio
import numpy as np

class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def validation_step(
        self, batch, batch_idx, dataloader_idx=0):
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
            
        Returns:
            The predicted mask.
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.val_metrics(y_hat_hard, y)
        self.log_dict(self.val_metrics)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"image/{batch_idx}", fig, global_step=self.global_step
                    )
                    plt.close()
            except ValueError:
                pass
        return y_hat_hard # return output for logging purposes