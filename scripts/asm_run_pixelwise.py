import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchgeo.datasets import NonGeoDataset, stack_samples, unbind_samples
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.trainers import BaseTask
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision.transforms.functional import pad
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import pytorch_lightning as pl
import geopandas as gpd
import rasterio
import numpy as np
import wandb

sys.path.append("/n/home07/kayan/asm/scripts/")
from asm_datamodules import *

# device configuration
device, num_devices = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", mp.cpu_count())
workers = mp.cpu_count()
print(f"Running on {num_devices} {device}(s) with {workers} cpus")

# model parameters
lr = 1e-3
n_epoch = 30
batch_size = 64
loss = "ce"
class_weights = [0.2,0.8]
num_workers=3
input_channels = 4
num_classes = 2
num_filters = 64
split = True
split_n = None
mines_only = True

# file names and paths
root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
#root = "/n/home07/kayan/asm/data/"
project = "ASM_seg_test" # project name in WandB
run_name = "2_perpositionmlp_all"

datamodule = ASMDataModule(batch_size=batch_size, num_workers=num_workers, split=split, split_n=split_n, root=root, transforms=min_max_transform, mines_only=mines_only)

class PerPositionMLP(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters):
        super().__init__()

        # 1x1 convolution
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class PerPixelClassificationTask(BaseTask):
    def __init__(self, loss, input_channels, num_classes, num_filters, class_weights=None, ignore_index=None,
                lr = 1e-3, patience = 10,):
        super(PerPixelClassificationTask, self).__init__(ignore="weights")
        
    def configure_losses(self):
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        if loss == "ce":
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=self.hparams["class_weights"]
            )
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )
    
    def configure_metrics(self):
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="micro"
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        
    def configure_models(self):
        self.model = PerPositionMLP(self.hparams["input_channels"], self.hparams["num_classes"], self.hparams["num_filters"])

    def training_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.train_metrics(y_hat_hard, y)
        self.log_dict(self.train_metrics)
        return loss
    
    def validation_step(
        self, batch, batch_idx, dataloader_idx=0):
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

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat_hard, y)
        self.log_dict(self.test_metrics)

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        y_hat = self(x).softmax(dim=1)
        return y_hat
    
task = PerPixelClassificationTask(
    loss=loss,
    class_weights = torch.Tensor(class_weights),
    input_channels=input_channels,
    num_classes=num_classes,
    num_filters=num_filters,
    lr=lr,
    patience=5
)

wandb_logger = WandbLogger(project=project, name=run_name, log_model="all")

class WandBCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # log train loss to WandB
        train_loss = trainer.callback_metrics.get("train_loss_epoch")
        if train_loss is not None:
            wandb.log({"train_loss": train_loss.item()}, step=trainer.global_step)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        # Log validation loss to WandB
        val_loss = trainer.callback_metrics.get("val_loss_epoch")
        if val_loss is not None:
            wandb.log({"val_loss": val_loss.item()}, step=trainer.global_step)
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            n = 4
            imgs = batch["image"]
            masks = batch["mask"].to(torch.float64)
            with torch.no_grad():
                predictions = pl_module(imgs)
            predictions = predictions.argmax(dim=1)
            predictions = predictions.to(torch.float64)
            captions = ["Image", "Ground truth", "Prediction"]
            for i in range(n):
                img = imgs[i][:-1] # remove NIR channel for plotting purposes
                wandb_logger.log_image(key=f"Train {batch_idx}-{i}", images=[img, masks[i], predictions[i]], caption=captions)
                    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
 
        # outputs corresponds to our model predictions
        if batch_idx == 0:
        # log n sample image predictions from first batch
            n = 4
            imgs = batch["image"]
            masks = batch["mask"].to(torch.float64)
            outputs = outputs.to(torch.float64)
            captions = ["Image", "Ground truth", "Prediction"]
            for i in range(n):
                img = imgs[i][:-1] # remove NIR channel for plotting purposes
                wandb_logger.log_image(key=f"Val {batch_idx}-{i}", images=[img, masks[i], outputs[i]], caption=captions)
                
trainer = Trainer(
        accelerator=device,
        devices=num_devices,
        max_epochs=n_epoch,
        callbacks=[WandBCallback()],
        logger=wandb_logger
    )

trainer.fit(model=task, datamodule=datamodule)
wandb.finish()