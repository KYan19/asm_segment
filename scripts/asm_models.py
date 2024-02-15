import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchgeo.datasets import NonGeoDataset, stack_samples, unbind_samples
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.trainers import PixelwiseRegressionTask, SemanticSegmentationTask
from torchgeo.models import RCF
from torchvision.transforms.functional import pad
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import geopandas as gpd
import rasterio
import numpy as np

class RCFRegression(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()

        # linear layer
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.linear(x)
        x = x.permute(0,3,1,2)
        return x

class CustomRCFModel(RCF):
    def forward(self, x):
        """Forward pass of the RCF model.

        Args:
            x: a tensor with shape (B, C, H, W)

        Returns:
            a tensor of size (B, ``self.num_features``)
        """
        # adjust padding to keep output same size as input
        x1a = F.relu(
            F.conv2d(x, self.weights, bias=self.biases, stride=1, padding="same"),
            inplace=True,
        )
        x1b = F.relu(
            -F.conv2d(x, self.weights, bias=self.biases, stride=1, padding="same"),
            inplace=False,
        )

        # skip the average pooling step from usual RCF model
        #x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        #x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:  # case where we passed a single input
            output = torch.cat((x1a, x1b), dim=0)
            return output
        else:  # case where we passed a batch of > 1 inputs
            output = torch.cat((x1a, x1b), dim=1)
            return output

class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]
        num_filters: int = self.hparams["num_filters"]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
            #for m in self.model.modules(): #FLAG
            #    if isinstance(m, nn.BatchNorm2d):
            #        m.track_running_stats=False
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        elif model == "rcf":
            self.model = RCFRegression(input_features=in_channels, num_classes=num_classes)
            
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if model in ["unet", "deeplabv3+"]:
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams["freeze_backbone"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
                
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
        return torch.softmax(y_hat,dim=1)[:,1] # return output for logging purposes
    
    def test_step(self, batch, batch_id, dataloader_idx = 0):
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        
        #if batch_id==0:
        #    print(f"Input data type: {x.dtype}")
        #    print(f"Example input: {x[0]}")
        #    print(f"Sum of each channel: {x[0].sum(axis=-1).sum(axis=-1)}")
        #    print(f"Example output: {y_hat_hard[0]}")
        
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat_hard, y)
        self.log_dict(self.test_metrics)
