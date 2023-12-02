import pickle
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split

def split_asm_data(
    path="/n/home07/kayan/data/filtered_labels.geojson", 
    stratify_col="country", 
    save = True,
    out_path = "/n/home07/kayan/data/train_test_split",
    n = None
):
    """Split data into train/test/val sets.

    Args:
        path: str, optional
            path to geojson file used to load dataframe. Default is '/n/home07/kayan/data/filtered_labels.geojson'
        stratify_col: str, optional
            the name of the column used to stratify the data. Default is 'country'.
        save: bool, optional
            whether or not to save the split in a pickle file. Default is True.
        out_path: str, optional
            path used to save file if save is True. Default is '/n/home07/kayan/data/train_test_split'
        n: int, optional
            restrict split to first n items in dataframe. Default is None.
    """  
    
    label_df = gpd.read_file(path)
    if n is not None:
        label_df = label_df.head(n)
    
    # split into train/val and test
    train, test = train_test_split(label_df, 
                stratify=label_df[stratify_col] if stratify_col is not None else None,
                test_size=0.2
            )
    # split further into train and val
    train, val = train_test_split(train,
                stratify=train[stratify_col] if stratify_col is not None else None,
                test_size=0.2)
                                  
    # get unique identifiers for each split
    train_ids = train["unique_id"].values
    val_ids = val["unique_id"].values
    test_ids = test["unique_id"].values
    print(f"Split with {len(train_ids)} train images, {len(val_ids)} validation images, and {len(test_ids)} test images")
    print(f"Mine proportions\n Train: {len(train[train['label']==1.0])/len(train)}")
    print(f" Validation: {len(val[val['label']==1.0])/len(val)}")
    print(f" Test: {len(test[test['label']==1.0])/len(test)}")
    split_ids = {"train": train_ids, "val": val_ids, "test":test_ids}
    
    if save:
        # save as pickle file
        with open(out_path, 'wb') as handle:
            pickle.dump(split_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path