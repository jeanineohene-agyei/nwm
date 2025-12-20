import os
import shutil
import pickle
import numpy as np

# -------- CONFIG --------
src_dir = "/home/jeanine/EPIC-KITCHENS/P01/videos/P01_107"   # folder with all JPGs
n_train = 200
n_test = 50

dest_root = "/home/jeanine/nwm/data/epic_kitchens"
train_name = "P01_107_train200"
test_name = "P01_107_test50"
# --------------------------------

# collect all frames
files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")])
print("Total frames found:", len(files))

train_files = files[:n_train]
test_files = files[n_train:n_train+n_test]

print(f"Train frames: {len(train_files)}")
print(f"Test frames:  {len(test_files)}")

# --- Function to create an NWM trajectory ---
def make_traj(folder_name, file_list):
    traj_dir = os.path.join(dest_root, folder_name)
    os.makedirs(traj_dir, exist_ok=True)

    # copy frames and rename to 0.jpg ... N.jpg
    for idx, fname in enumerate(file_list):
        shutil.copy(
            os.path.join(src_dir, fname),
            os.path.join(traj_dir, f"{idx}.jpg")
        )

    # NWM-required metadata
    num_frames = len(file_list)
    traj_data = {
        "position": np.zeros((num_frames, 2), dtype=float),
        "yaw": np.zeros((num_frames,), dtype=float),
    }

    with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f)

    print(f"Created trajectory: {traj_dir}")

# make train + test trajectories
make_traj(train_name, train_files)
make_traj(test_name, test_files)

print("Done!")
