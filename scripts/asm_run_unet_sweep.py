import os
import sys
import random
import shutil
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import NonGeoDataset, stack_samples, unbind_samples
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.trainers import PixelwiseRegressionTask, SemanticSegmentationTask
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
from torchvision.transforms.functional import pad
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import geopandas as gpd
import rasterio
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, auc
from asm_datamodules import *
from asm_models import *

# PARAMETERS
split = False # generate new splits if True; use saved splits if False
project = "ASM_seg_sweep" # project name in WandB
run_name = "2_sweep" # run name in WandB
n_epoch = 30
class_weights = None

# ---SET UP HYPERPARAMETER SWEEP---
# configuration
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "lr": {"values": [1e-3, 1e-4, 1e-5]},
        "weight_decay": {"values": [1e-1, 1e-2, 1e-3]},
        "loss": {"value": "focal"}
    }
}
# set up sweep in WandB
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

# device configuration
device, num_devices = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", len(os.sched_getaffinity(0)))
workers = len(os.sched_getaffinity(0))
print(f"Running on {num_devices} {device}(s) with {workers} cpus")
torch.set_float32_matmul_precision("medium")

# ---PREPARE DATA---
if split:
    # Set the random seed for reproducibility
    seed = 42
    random.seed(seed)

    # Define the number of random splits to perform
    n_splits = 10

    # Generate n_splits random integers between 0 and 1,000,000
    random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]
    
    for random_state in random_seeds:
        # unique file path for this split
        out_path = "/n/home07/kayan/asm/data/splits/split_"+str(random_state)
        
        # generate split and save as pickle file
        split_asm_data(stratify_col="country", 
                       save = True,
                       out_path=out_path, 
                       random_state=random_state)

# datamodule configuration
batch_size = 64
num_workers = workers # set to maximum number of available CPUs
split = False # use a saved split instead of generating new one
root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
split_path = "/n/home07/kayan/asm/data/splits/split_670487" # one of the randomly generated splits

# create and set up datamodule
datamodule = ASMDataModule(batch_size=batch_size, 
                           num_workers=num_workers, 
                           split=split, 
                           root=root, 
                           transforms=min_max_transform, 
                           split_path=split_path)

# set up dataloaders
datamodule.setup("fit")
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader() 
datamodule.setup("test")
test_dataloader = datamodule.test_dataloader()

# callback for WandB logging
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
        if batch_idx%4 == 0:
            n = 1
            imgs = batch["image"]
            masks = batch["mask"].to(torch.float64)
            outputs = outputs.to(torch.float64)
            captions = ["Image", "Ground truth", "Prediction"]
            for i in range(n):
                img = ToPILImage()(imgs[i][:-1]) # remove NIR channel for plotting purposes
                mask = ToPILImage()(masks[i])
                output = ToPILImage()(outputs[i])
                wandb.log({f"Val {batch_idx}-{i}": [wandb.Image(image) for image in [img, mask, output]]})
                #wandb_logger.log_image(key=f"Val {batch_idx}-{i}", images=[img, mask, output], caption=captions)
    
def main():
    wandb.init(name=run_name, project=project)
    wandb_logger = WandbLogger(project=project, name=run_name, log_model=True, save_code=True)
    
    # get hyperparameter values 
    lr = wandb.config["lr"]
    weight_decay = wandb.config["weight_decay"]
    loss = wandb.config["loss"]
    
    # set up model
    task = CustomSemanticSegmentationTask(
        model="unet",
        backbone="resnet18",
        weights=True, # use ImageNet weights
        loss=loss,
        class_weights = torch.Tensor(class_weights) if class_weights is not None else None,
        in_channels=4,
        num_classes=2,
        lr=lr,
        weight_decay=weight_decay,
        freeze_backbone=False,
        freeze_decoder=False
    )
    
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    early_stop_callback = EarlyStopping(
           monitor='val_loss',
           min_delta=0.00,
           patience=3,
           verbose=False,
           mode='min'
        )

    trainer = Trainer(
            accelerator=device,
            devices=num_devices,
            max_epochs=n_epoch,
            callbacks=[WandBCallback(), checkpoint_callback, early_stop_callback],
            logger=wandb_logger
        )

    trainer.fit(model=task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=task, dataloaders=test_dataloader)
    wandb.finish()
    
wandb.agent(sweep_id, function=main)