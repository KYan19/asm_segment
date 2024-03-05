import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchgeo.datasets import NonGeoDataset
from torchgeo.datamodules import NonGeoDataModule
from torchvision.transforms import Resize, InterpolationMode, RandomCrop
from torchvision.transforms.functional import pad
import geopandas as gpd
import rasterio
import numpy as np
from asm_train_test_split import split_asm_data
from asm_models import *

def min_max_transform(sample, target_size=(256,256), crop_size=None):
    img = sample["image"].permute(1, 2, 0) # moves spectral channels to final dimension
    mask = sample["mask"]
    
    # min-max scaling by channel
    img = img.numpy()
    img_norm = (img - np.min(img,axis=(0,1))) / (np.max(img,axis=(0,1)) - np.min(img,axis=(0,1))) 
    img_norm = torch.tensor(img_norm).permute(2, 0, 1) # re-permute spectral channels to first dimension
    
    # resize data to be 256x256
    img_norm = Resize((256,256),antialias=True)(torch.unsqueeze(img_norm,0))
    mask = Resize((256,256),interpolation=InterpolationMode.NEAREST)(torch.unsqueeze(mask, 0))
    
    if crop_size is not None:
        crop_transform = RandomCrop(crop_size)
        img_norm = crop_transform(img_norm)
        mask = crop_transform(mask)
    
    sample["image"] = torch.squeeze(img_norm)
    sample["mask"] = torch.squeeze(mask)
    return sample

def rcf(sample, in_channels = 4, features = 1000, kernel_size = 3, bias = -1.0, seed=42, crop_size=32, **kwargs):
    # first normalize input
    norm_sample = min_max_transform(sample, crop_size=crop_size)
    norm_sample["norm_image"] = norm_sample["image"] # save normalized image separately
    # extract RCF features, output size (B, features, crop_size, crop_size)
    rcf_model = CustomRCFModel(in_channels=in_channels, features=features, kernel_size=kernel_size, bias=bias, seed=seed, **kwargs)
    img = norm_sample["image"].unsqueeze(dim=0)
    norm_sample["image"] = rcf_model(img).squeeze()
    return norm_sample

class ASMDataset(NonGeoDataset):
    splits = ["train", "val", "test"]
    all_bands = ["B", "G", "R", "NIR", "Mask"]
    rgb_bands = ["R", "G", "B"]
    
    def __init__(
        self,
        root = "/n/holyscratch01/tambe_lab/kayan/karena/",
        transforms = None,
        split = "train",
        bands = ["R", "G", "B", "NIR"],
        split_path = "/n/home07/kayan/asm/data/train_test_split",
        **kwargs
    ) -> None:
        """Initialize a new ASMData instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            split: one of "train," "val", or "test"
            bands: the subset of bands to load
            split_path: path to file containing unique identifiers for train/test/val split, generated with scripts/train_test_split.py
            **kwargs: Additional keyword arguments passed on to transform function
        """  
        self.root = root
        self.transforms = transforms
        self.transform_args = kwargs
        assert split in ["train", "val", "test"]
        self.bands = bands
        self.band_indices = [self.all_bands.index(b) + 1 for b in bands if b in self.all_bands] # add 1 since rasterio starts index at 1, not 0
        
        # get unique identifiers of desired split
        with open(split_path,'rb') as handle:
            split_data = pickle.load(handle)
        self.ids = split_data[split]
        
        # convert unique identifiers to file names
        self.image_filenames = [f"{self.root}images/{unique_id}.tif" for unique_id in self.ids]
        self.mask_filenames = [f"{self.root}rasters/{unique_id}.tif" for unique_id in self.ids]
        
    def __len__(self):
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.image_filenames)
        
    def __getitem__(self, index: int):
        """Return item at an index within the dataset.

        Args:
            index: index to return

        Returns:
            a dict containing image, mask, transform, crs, and metadata at index.
        """
        img_fn = self.image_filenames[index]
        mask_fn = self.mask_filenames[index]
        
        img = rasterio.open(img_fn).read(self.band_indices)
        img = torch.from_numpy(np.array(img, dtype=np.float32))
        
        mask = rasterio.open(mask_fn).read(1)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        sample = {"image": img, "mask": mask, "id": self.ids[index]}

        if self.transforms is not None:
            sample = self.transforms(sample, **self.transform_args)
            
        return sample
    
    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.bands.index(band))

        # Reorder and rescale the image
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = image.numpy()
        # min-max scaling
        #image_norm = (image - np.min(image,axis=(0,1))) / (np.max(image,axis=(0,1)) - np.min(image,axis=(0,1))) 
        
        # Reorder mask
        mask = sample["mask"]

        # Plot the image
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(image)
        axs[1].imshow(mask,cmap="gray")
        axs[0].axis("off")
        axs[0].set_title("Image")
        axs[1].axis("off")
        axs[1].set_title("Mask")

        return fig
    
class ASMDataModule(NonGeoDataModule):
    def __init__(
        self, 
        batch_size: int = 8, 
        num_workers: int = 1,
        split: bool = False,
        split_n: int = None,
        save: bool = False,
        mines_only: bool = False,
        **kwargs
    ) -> None:
        """Initialize a new ASMModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            split: Whether or not to perform a new train-test-val split of data.
            split_n: Number of tiles to include in train-test-val split
            mines_only: restrict data to only images that have mines in them
            **kwargs: Additional keyword arguments passed to ASMDataset.
        """
        # perform train-test-val split and pass path to output file as kwarg to ASMDataset
        if split:
            split_path = split_asm_data(n=split_n, save=save, mines_only=mines_only)
            kwargs["split_path"] = split_path
        super().__init__(ASMDataset, batch_size, num_workers, **kwargs)
        
def get_conv_dataloader(pixelwise_predictions, 
                     batch_size,
                     label_path="/n/home07/kayan/asm/data/filtered_labels.geojson", 
                     split="train"):
    assert split in ["train", "val", "test"]
    inputs = np.array(list(pixelwise_predictions.values()))
    label_df = gpd.read_file(label_path)
    labels = [label_df[label_df["unique_id"]==x]["label"].values[0] for x in pixelwise_predictions.keys()]
    
    tensor_inputs = torch.Tensor(inputs)
    tensor_inputs = tensor_inputs[:, None, :, :]
    tensor_labels = torch.LongTensor(labels)
    
    shuffle = True if split=="train" else False
    dataset = TensorDataset(tensor_inputs,tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader