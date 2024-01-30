import sys
import shutil
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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from asm_datamodules import *
from asm_models import *

# device configuration
device, num_devices = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", mp.cpu_count())
workers = mp.cpu_count()
torch.set_num_threads(8)
print(f"Running on {num_devices} {device}(s) with {workers} cpus")
print(f"Torch indicates there are {torch.get_num_threads()} CPUs")

# model parameters
lr = 1e-5
n_epoch = 20
batch_size = 64
loss = "ce"
class_weights = [0.2,0.8]
num_workers = 8
mines_only = False
split = False
split_n = None
split_path = "/n/home07/kayan/asm/data/splits/9_all_data_lowlr_save-split"
freeze_backbone = False
save_split = False

# file names and paths
root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
#root = "/n/home07/kayan/asm/data/"
project = "ASM_seg" # project name in WandB
run_name = "13_all_data_lowlr_dataloader_w/eval"

# create dataloaders
train_dataset = ASMDataset(root = root, transforms = min_max_transform, split = "train", 
                           bands = ["R", "G", "B", "NIR"], split_path = split_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)

val_dataset = ASMDataset(root = root, transforms = min_max_transform, split = "val",
                          bands = ["R", "G", "B", "NIR"], split_path = split_path)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

test_dataset = ASMDataset(root = root, transforms = min_max_transform, split = "test",
                          bands = ["R", "G", "B", "NIR"], split_path = split_path)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

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

wandb_logger = WandbLogger(project=project, name=run_name, log_model="all", save_code=True)
checkpoint_callback = ModelCheckpoint(every_n_epochs=1)

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
        callbacks=[WandBCallback(), checkpoint_callback],
        logger=wandb_logger
    )

trainer.fit(model=task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
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
        # average probability in pixels classified as mine (>0.5)
        class_proba[unique_id] = np.mean(preds*(preds>0.5))
        #class_proba[unique_id] = np.sum((preds>0.5))
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
    
for name, param in task.named_parameters():
    if 'encoder.layer4.1.bn2' in name:  # Check if the parameter belongs to a batch normalization layer
        print(f'Parameter name: {name}, Value: {param}')
