import os
import h5py
import numpy as np
from tqdm import tqdm

def copy_dataset(split):
    root_dir = "/mnt/e/EVE_dataset_processed_error"
    participant_list = os.listdir(os.path.join(root_dir, split))
    participant_path_list = [os.path.join(root_dir, split, participant) for participant in participant_list]

    dest_dir = "/mnt/e/EVE_dataset_processed"
    dest_files = {}
    dest_path_list = [os.path.join(dest_dir, split, participant) for participant in participant_list]

    for i, dest_path in enumerate(tqdm(dest_path_list)):
        if not os.path.exists(os.path.dirname(dest_path)):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # dest_files[i] = h5py.File(dest_path, "w")

    for i, participant_path in enumerate(tqdm(participant_path_list)):
        with h5py.File(participant_path, "r") as src:
            total_rows = src["images"].shape[0]
            with h5py.File(dest_path_list[i], "w") as dest:
                dest.create_dataset("images", data=src["images"][3000:total_rows], dtype=np.float32)
                dest.create_dataset("labels", data=src["labels"][3000:total_rows], dtype=np.float32)

if __name__ == '__main__':
    copy_dataset("train")
    copy_dataset("val")
    copy_dataset("test")