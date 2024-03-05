import os
import sys
import random
import shutil
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
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
project = "ASM_conv_sweep" # project name in WandB
run_name = "1_sweep" # run name in WandB
n_epoch = 10
class_weights = [1/6,5/6]

# ---SET UP HYPERPARAMETER SWEEP---
# configuration
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "lr": {"values": [1e-2, 1e-3, 1e-4]},
        "weight_decay": {"values": [1e-1, 1e-2, 1e-3]}
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
data_dir = "/n/home07/kayan/asm/data/"
suffix = "_focal"
with open(data_dir+"train_seg_preds"+suffix, 'rb') as handle:
    train_pixelwise_predictions = pickle.load(handle)
with open(data_dir+"val_seg_preds"+suffix, 'rb') as handle:
    val_pixelwise_predictions = pickle.load(handle)
with open(data_dir+"test_seg_preds"+suffix, 'rb') as handle:
    test_pixelwise_predictions = pickle.load(handle)

# datamodule configuration
batch_size = 64
num_workers = workers # set to maximum number of available CPUs

# set up dataloaders
train_dataloader = get_conv_dataloader(train_pixelwise_predictions, batch_size=batch_size, split="train")
val_dataloader = get_conv_dataloader(val_pixelwise_predictions, batch_size=batch_size, split="val")
test_dataloader = get_conv_dataloader(test_pixelwise_predictions, batch_size=batch_size, split="test")

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
    
    # set up model
    model = LightningConvNet(class_weights=class_weights, lr=lr, weight_decay=weight_decay)
    
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

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)
    
    class_proba = []
    true_labels = []
    for i, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        outputs = torch.softmax(model.to(device)(images),dim=-1).cpu().detach()
        class_proba.extend(outputs[:,1].tolist())
        true_labels.extend(labels.tolist())
        
    fig,ax = plt.subplots()
    fpr, tpr, _ = roc_curve(true_labels, class_proba)
    ax.plot(fpr, tpr)
    ax.set_title("Test ROC")
    ax.set_xlabel("False Positive")
    ax.set_ylabel("True Positive")
    wandb.log({"ROC": wandb.Image(fig)})
    
    auc = roc_auc_score(true_labels, class_proba)
    print(f"lr: {lr}, weight decay: {weight_decay}, AUC: {auc}")
    
    wandb.finish()
    
wandb.agent(sweep_id, function=main)
