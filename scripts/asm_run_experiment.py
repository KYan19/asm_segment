import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import NonGeoDataset, stack_samples, unbind_samples
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.trainers import PixelwiseRegressionTask, SemanticSegmentationTask
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms.functional import pad
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import geopandas as gpd
import rasterio
import numpy as np
import wandb
from asm_datamodules import *
from asm_models import *

# device configuration
device, num_devices = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", mp.cpu_count())
workers = mp.cpu_count()
print(f"Running on {num_devices} {device}(s) with {workers} cpus")

# model parameters
lr = 1e-5
n_epoch = 20
batch_size = 64
loss = "ce"
class_weights = [0.2,0.8]
num_workers = 8
mines_only = False
split = True
split_n = None
freeze_backbone = False

# file names and paths
root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
#root = "/n/home07/kayan/asm/data/"
project = "ASM_seg" # project name in WandB
run_name = "8_all_data_lowlr"

datamodule = ASMDataModule(batch_size=batch_size, num_workers=num_workers, split=split, split_n=split_n, root=root, transforms=min_max_transform, mines_only=mines_only)

task = CustomSemanticSegmentationTask(
    model="unet",
    backbone="resnet18",
    weights=True,
    loss=loss,
    class_weights = torch.Tensor(class_weights),
    in_channels=4,
    num_classes=2,
    lr=lr,
    patience=5,
    freeze_backbone=freeze_backbone,
    freeze_decoder=False
)

wandb_logger = WandbLogger(project=project, name=run_name, log_model="true")

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
                    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
 
        # outputs corresponds to our model predictions
        # log n sample image predictions from every other batch
        if batch_idx%2 == 0:
            n = 1
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
