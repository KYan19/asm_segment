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

sys.path.append("/n/home07/kayan/asm/scripts/")
from asm_datamodules import *
from asm_models import *

# PARAMETERS
split = False # generate new splits if True; use saved splits if False
project = "ASM_seg_global" # project name in WandB
run_name = "unet_global_split670487" # run name in WandB
n_epoch = 30
class_weights = None
alpha = 0.75
gamma = 1
lr = 1e-3
weight_decay = 0.01
loss = "focal"
patience = 2
early_stop_patience = 10
batch_size = 64

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
num_workers = workers # set to maximum number of available CPUs
root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
split_path = "/n/home07/kayan/asm/data/splits/split_670487" # one of the randomly generated splits
#split_path = "/n/home07/kayan/asm/data/splits/split_LOCO_TZA"

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
                #wandb.log({f"Val {batch_idx}-{i}": [wandb.Image(image) for image in [img, mask, output]]})
                wandb_logger.log_image(key=f"Val {batch_idx}-{i}", images=[img, mask, output], caption=captions)

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
    alpha=alpha,
    gamma=gamma,
    patience=patience,
    freeze_backbone=False,
    freeze_decoder=False
)

wandb_logger = WandbLogger(project=project, name=run_name, log_model=True, save_code=True)
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
    
# set up datamodule for testing
datamodule.setup("test")
test_dataloader = datamodule.test_dataloader()

trainer.test(model=task, dataloaders=test_dataloader)

# construct ROC curve
task.eval()
pixelwise_predictions = {}

with torch.inference_mode():
    for idx,samples in enumerate(test_dataloader):
        unique_ids = samples['id']
        # Move input data to the device
        inputs = samples['image']

        # Forward pass
        outputs = task(inputs)
        outputs = torch.softmax(outputs, dim=1)
        #outputs = outputs.argmax(dim=1).squeeze()
        
        for unique_id,output in zip(unique_ids, outputs):
            pixelwise_predictions[unique_id] = output[1].cpu().numpy()
            
def pixelwise_to_class(pixelwise_preds):
    class_proba = {}
    for (unique_id,preds) in pixelwise_preds.items():
        # average probability across pixels
        class_proba[unique_id] = np.mean(preds)
    return class_proba

class_proba = pixelwise_to_class(pixelwise_predictions)
path="/n/home07/kayan/asm/data/filtered_labels.geojson"
label_df = gpd.read_file(path)

true_labels = [label_df[label_df["unique_id"]==x]["label"].values[0] for x in class_proba.keys()]
class_proba = list(class_proba.values())
fig,ax = plt.subplots()
fpr, tpr, _ = roc_curve(true_labels, class_proba)
ax.plot(fpr, tpr)
ax.set_title("Test ROC")
ax.set_xlabel("False Positive")
ax.set_ylabel("True Positive")
wandb.log({"ROC": wandb.Image(fig)})

auc = roc_auc_score(true_labels, class_proba)
wandb.log({"Test AUC": auc})
print(f"AUC is: {auc}")

wandb.finish()
