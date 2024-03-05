import os
from pathlib import Path
import pickle
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split

def split_asm_data(
    path="/n/home07/kayan/asm/data/filtered_labels.geojson", 
    data_path="/n/holyscratch01/tambe_lab/kayan/karena/images/",
    stratify_col="country", 
    save = True,
    out_path = "/n/home07/kayan/asm/data/train_test_split",
    n = None,
    mines_only = False,
    random_state = None
):
    """Split data into train/test/val sets.

    Args:
        path: str, optional
            path to geojson file used to load dataframe. Default is '/n/home07/kayan/data/filtered_labels.geojson'
        data_path: str, optional
            path to directory with image files, used to cross-reference unique_ids in geojson file. Default is '/n/holyscratch01/tambe_lab/kayan/karena/images/'
        stratify_col: str, optional
            the name of the column used to stratify the data. Default is 'country'.
        save: bool, optional
            whether or not to save the split in a pickle file. Default is True.
        out_path: str, optional
            path used to save file if save is True. Default is '/n/home07/kayan/data/train_test_split'
        n: int, optional
            restrict split to first n items in dataframe. Default is None.
        mines_only: bool, optional
            restrict data to only images that have mines in them. Default is False.
        random-state: int, optional
            random seed used to generate train-test split. Default is None.
    """  
    
    label_df = gpd.read_file(path)
    if n is not None:
        label_df = label_df.head(n)
        
    if mines_only: 
        label_df = label_df[label_df["label"] == 1]
        
    # take out any files that are not present in the image directory
    dir_ids = [Path(file_name).stem for file_name in os.listdir(data_path)]
    label_df = label_df[label_df["unique_id"].isin(dir_ids)]
    
    # split into train/val and test
    train, test = train_test_split(label_df, 
                stratify=label_df[stratify_col] if stratify_col is not None else None,
                test_size=0.2,
                random_state=random_state
            )
    # split further into train and val
    train, val = train_test_split(train,
                stratify=train[stratify_col] if stratify_col is not None else None,
                test_size=0.2,
                random_state=random_state)
                                  
    # get unique identifiers for each split
    train_ids = train["unique_id"].values
    val_ids = val["unique_id"].values
    test_ids = test["unique_id"].values
    
    split_ids = {"train": train_ids, "val": val_ids, "test":test_ids}
    if save:
        # save as pickle file
        with open(out_path, 'wb') as handle:
            pickle.dump(split_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path