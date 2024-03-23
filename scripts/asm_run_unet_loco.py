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
from torchmetrics import AUROC
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import geopandas as gpd
import rasterio
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, auc
from asm_datamodules import *
from asm_models import *

# callback to calculate val AUC at end of each epoch
# uses a simple global averaging function to transform seg output -> image pred
class AUCCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_dataloader = trainer.val_dataloaders
        
        pl_module.eval()
        preds = []
        targets = []
        with torch.inference_mode():
            for samples in val_dataloader:
                inputs = samples["image"].to(device)
                masks = samples["mask"].to(device)
                
                outputs = pl_module(inputs) # get model output
                outputs = torch.softmax(outputs, dim=1) # softmax over class dimension
                
                img_preds = torch.mean(outputs, dim=(-1,-2)) # average over x and y dimensions
                img_targets = (torch.sum(masks, dim=(-1,-2)) > 0) # will be >0 if contains a mine, 0 otherwise
                preds.append(img_preds[:,1]) # append probability of mine class
                targets.append(img_targets) # append true labels
                
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        auc_task = AUROC(task="binary")
        auc_score = auc_task(preds,targets)
        
        wandb.log({"val_AUC": auc_score.item()}, step=trainer.global_step)

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
    
def main():
    wandb.init(name=run_name, project=project)
    wandb_logger = WandbLogger(project=project, name=run_name, log_model=True, save_code=True)
    
    # get hyperparameter values 
    lr = wandb.config["lr"]
    weight_decay = wandb.config["weight_decay"]
    loss = wandb.config["loss"]
    alpha = wandb.config["alpha"]
    gamma = wandb.config["gamma"]
    
    # set up model
    task = CustomSemanticSegmentationTask(
        model="unet",
        backbone="resnet18",
        weights=True, # use ImageNet weights
        loss=loss,
        class_weights = torch.Tensor(class_weights) if class_weights is not None else None,
        alpha=alpha,
        gamma=gamma,
        in_channels=4,
        num_classes=2,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        freeze_backbone=False,
        freeze_decoder=False
    )
    
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    early_stop_callback = EarlyStopping(
           monitor='val_loss',
           min_delta=0.00,
           patience=early_stop_patience,
           verbose=False,
           mode='min'
        )
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        accelerator=device,
        devices=num_devices,
        max_epochs=n_epoch,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[AUCCallback(), WandBCallback(), checkpoint_callback, early_stop_callback, lr_callback],
        logger=wandb_logger
    )

    trainer.fit(model=task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    #trainer.test(model=task, dataloaders=test_dataloader)
    wandb.finish()

countries = ['SLE', 'COD', 'CAF', 'ZWE', 'TZA']

for i, country in enumerate(countries):
    # PARAMETERS
    split = False # generate new splits if True; use saved splits if False
    project = "ASM_seg_sweep" # project name in WandB
    run_name = f"{7+i}_loco_sweep_{country}" # run name in WandB
    n_epoch = 20
    class_weights = None
    patience = 3
    early_stop_patience = 8

    # ---SET UP HYPERPARAMETER SWEEP---
    # configuration
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "lr": {"values": [1e-3, 1e-4]},
            "alpha": {"values": [3/4, 7/8]}, # focal loss hparam
            "gamma": {"values": [1, 2]}, # focal loss hparam
            "weight_decay": {"value": 1e-2},
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

    # datamodule configuration
    batch_size = 64
    num_workers = workers # set to maximum number of available CPUs
    split = False # use a saved split instead of generating new one
    root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
    split_path = f"/n/home07/kayan/asm/data/splits/split_LOCO_{country}" # one of the randomly generated splits

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

    wandb.agent(sweep_id, function=main)