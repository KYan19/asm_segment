import sys
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
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import geopandas as gpd
import rasterio
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, auc
from asm_datamodules import *
from asm_models import *

# device configuration
device, num_devices = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", mp.cpu_count())
workers = mp.cpu_count()
print(f"Running on {num_devices} {device}(s) with {workers} cpus")
print(f"Torch indicates there are {torch.get_num_threads()} CPUs")

# model parameters
lr = 1e-4
n_epoch = 10
batch_size = 64
loss = "ce"
class_weights = [0.05,0.95]
num_workers = 4
mines_only = False
split = False
split_n = None
split_path = "/n/home07/kayan/asm/data/splits/9_all_data_lowlr_save-split"
freeze_backbone = False
save_split = False

# file names and paths
root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
project = "ASM_seg" # project name in WandB
run_name = "21_rcf_heavy_class_weights"

# create and set up datamodule
datamodule = ASMDataModule(batch_size=batch_size, num_workers=num_workers, split=split, split_n=split_n, 
                           root=root, transforms=rcf, mines_only=mines_only, split_path=split_path)
datamodule.setup("fit")
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()

task = CustomSemanticSegmentationTask(
    model="rcf",
    weights=True,
    loss=loss,
    class_weights = torch.Tensor(class_weights) if class_weights is not None else None,
    in_channels=16,
    num_classes=2,
    lr=lr
)

wandb_logger = WandbLogger(project=project, name=run_name, log_model=True, save_code=True)

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
            imgs = batch["norm_image"]
            masks = batch["mask"].to(torch.float64)
            outputs = outputs.to(torch.float64)
            captions = ["Image", "Ground truth", "Prediction"]
            for i in range(n):
                img = ToPILImage()(imgs[i][:-1]) # remove NIR channel for plotting purposes
                mask = ToPILImage()(masks[i])
                output = ToPILImage()(outputs[i])
                wandb_logger.log_image(key=f"Val {batch_idx}-{i}", images=[img, mask, output], caption=captions)

trainer = Trainer(
        accelerator=device,
        devices=num_devices,
        max_epochs=n_epoch,
        callbacks=[WandBCallback()],
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
print(f"AUC is: {roc_auc_score(true_labels, class_proba)}")

wandb.finish()

if save_split:
    shutil.copyfile("/n/home07/kayan/asm/data/train_test_split", f"/n/home07/kayan/asm/data/splits/{run_name}-split")
