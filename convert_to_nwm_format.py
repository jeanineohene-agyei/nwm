import os
import shutil
import pickle
import numpy as np

# -------- CONFIG --------
src_dir = "/home/jeanine/nwm/EPIC-KITCHENS/P01_106"  # folder with JPGs
data_root = "/home/jeanine/nwm/data/epic_kitchens"
split_root = "/home/jeanine/nwm/data_splits/epic_kitchens"

traj_len = 12
stride = 2
train_ratio = 0.85
# --------------------------------

os.makedirs(data_root, exist_ok=True)
os.makedirs(split_root, exist_ok=True)

# collect all frames
files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")])
num_frames = len(files)

print("Total frames found:", num_frames)

traj_names = []
traj_count = 0

# -------- create trajectories --------
for start in range(0, num_frames - traj_len + 1, stride):
    traj_files = files[start : start + traj_len]

    traj_name = f"traj_{traj_count:05d}"
    traj_dir = os.path.join(data_root, traj_name)
    os.makedirs(traj_dir, exist_ok=True)

    # copy frames and rename to 0.jpg ... 11.jpg
    for idx, fname in enumerate(traj_files):
        shutil.copy(
            os.path.join(src_dir, fname),
            os.path.join(traj_dir, f"{idx}.jpg")
        )

    # NWM-required metadata
    traj_data = {
        "position": np.zeros((traj_len, 2), dtype=float),
        "yaw": np.zeros((traj_len,), dtype=float),
    }

    with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f)

    traj_names.append(traj_name)
    traj_count += 1

print(f"Created {traj_count} trajectories")

# -------- create splits --------
num_trajs = len(traj_names)
num_train = int(train_ratio * num_trajs)

train_trajs = traj_names[:num_train]
test_trajs = traj_names[num_train:]

train_dir = os.path.join(split_root, "train")
test_dir = os.path.join(split_root, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
    for name in train_trajs:
        f.write(name + "\n")

with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
    for name in test_trajs:
        f.write(name + "\n")

print(f"Train trajectories: {len(train_trajs)}")
print(f"Test trajectories:  {len(test_trajs)}")
print("Done!")
