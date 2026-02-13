# import numpy as np
# import torch
# from collections import defaultdict
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# # -----------------------------
# # 1) Load dataset
# # -----------------------------
# REPO_ID = "lerobot/xvla-soft-fold"
# ds = LeRobotDataset(REPO_ID)

# print("Dataset root:", ds.root)
# print("Total frames:", len(ds))

# # -----------------------------
# # 2) Group frames by episode
# # -----------------------------
# episode_frames = defaultdict(list)

# for i in range(50):
#     sample = ds[i]
#     ep_id = sample["episode_index"].item()
#     episode_frames[ep_id].append(i)

# print("Number of episodes:", len(episode_frames))

# # -----------------------------
# # 3) Select one episode
# # -----------------------------
# ep_id = list(episode_frames.keys())[0]
# indices = episode_frames[ep_id]

# print(f"\nInspecting Episode {ep_id}")
# print("Number of frames in episode:", len(indices))

# # -----------------------------
# # 4) Stack full state trajectory
# # -----------------------------
# states = []
# timestamps = []

# for idx in indices:
#     sample = ds[idx]
#     states.append(sample["observation.state"].numpy())
#     timestamps.append(sample["timestamp"].item())

# states = np.stack(states)  # shape [T, 96]
# timestamps = np.array(timestamps)

# print("\nStates shape:", states.shape)

# # -----------------------------
# # 5) Basic statistics
# # -----------------------------
# print("\nGlobal state stats:")
# print("Min:", states.min())
# print("Max:", states.max())

# # -----------------------------
# # 6) Inspect Left Arm Euler
# # -----------------------------
# # Assumed layout:
# # eef_euler_0: left_x
# # eef_euler_1: left_y
# # eef_euler_2: left_z
# # eef_euler_3: left_roll
# # eef_euler_4: left_pitch
# # eef_euler_5: left_yaw

# left_x = states[:, 0]
# left_y = states[:, 1]
# left_z = states[:, 2]
# left_roll = states[:, 3]
# left_pitch = states[:, 4]
# left_yaw = states[:, 5]

# print("\nLeft arm position range:")
# print("x:", left_x.min(), "to", left_x.max())
# print("y:", left_y.min(), "to", left_y.max())
# print("z:", left_z.min(), "to", left_z.max())

# print("\nLeft arm rotation range (radians):")
# print("roll:", left_roll.min(), "to", left_roll.max())
# print("pitch:", left_pitch.min(), "to", left_pitch.max())
# print("yaw:", left_yaw.min(), "to", left_yaw.max())

# # -----------------------------
# # 7) Check Euler discontinuity risk
# # -----------------------------
# print("\nChecking for ±pi boundary crossing...")

# if left_yaw.min() < -3.0 or left_yaw.max() > 3.0:
#     print("⚠️ Yaw reaches near ±π. Discontinuity possible.")
# else:
#     print("✅ Yaw stays safely away from ±π.")

# # -----------------------------
# # 8) Optional: compute wrapped yaw delta
# # -----------------------------
# def angle_diff(a, b):
#     diff = a - b
#     return (diff + np.pi) % (2*np.pi) - np.pi

# yaw_deltas = angle_diff(left_yaw[1:], left_yaw[:-1])

# print("\nYaw delta stats:")
# print("Min delta:", yaw_deltas.min())
# print("Max delta:", yaw_deltas.max())

# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# # ======================
# # CONFIG
# # ======================
# REPO_ID = "lerobot/xvla-soft-fold"
# OUT_ROOT = "/home/jeanine/nwm/xvla_soft_fold"

# CAMERA_KEY = "observation.images.cam_left_wrist"

# ORIG_FPS = 20
# TARGET_FPS = 4
# STRIDE = ORIG_FPS // TARGET_FPS  # 5

# MAX_EPISODES = 10

# OUT_W, OUT_H = 320, 240

# os.makedirs(OUT_ROOT, exist_ok=True)

# # ======================
# # LOAD DATASET
# # ======================
# print("Loading dataset...")
# ds = LeRobotDataset(REPO_ID)
# print("Total samples:", len(ds))

# # ======================
# # STREAM EPISODES (4 FPS)
# # ======================
# current_episode = None
# episode_count = 0

# t_in = 0
# t_out = 0

# pose_buffer = []   # <-- will store [x, y, yaw]
# traj_dir = None

# for sample in tqdm(ds, desc="Extracting 4FPS episodes"):

#     ep_id = int(sample["episode_index"])

#     # Stop after 10 episodes
#     if current_episode is not None and ep_id != current_episode:
#         episode_count += 1
#         if episode_count >= MAX_EPISODES:
#             break

#     # ------------------
#     # New episode detected
#     # ------------------
#     if current_episode is None or ep_id != current_episode:

#         # Save previous episode pose file
#         if traj_dir is not None and len(pose_buffer) > 0:
#             pose_arr = np.array(pose_buffer, dtype=np.float32)
#             np.save(os.path.join(traj_dir, "pose_4fps.npy"), pose_arr)

#         # Reset
#         current_episode = ep_id
#         traj_dir = os.path.join(OUT_ROOT, f"episode_{ep_id:06d}")
#         os.makedirs(traj_dir, exist_ok=True)

#         t_in = 0
#         t_out = 0
#         pose_buffer = []

#         print(f"\nStarted episode {ep_id}")

#     # ------------------
#     # Downsample to 4 FPS
#     # ------------------
#     if t_in % STRIDE != 0:
#         t_in += 1
#         continue

#     # ------------------
#     # IMAGE
#     # ------------------
#     img = sample[CAMERA_KEY]

#     img = img.permute(1, 2, 0).cpu().numpy()
#     img = np.clip(img, 0.0, 1.0)
#     img = (img * 255.0).astype(np.uint8)

#     img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
#     img = img[:, :, ::-1]  # RGB → BGR

#     cv2.imwrite(os.path.join(traj_dir, f"{t_out}.jpg"), img)

#     # ------------------
#     # LEFT ARM POSE
#     # ------------------
#     state = sample["observation.state"].cpu().numpy()

#     left_x = state[0]
#     left_y = state[1]
#     left_yaw = state[5]

#     pose_buffer.append([left_x, left_y, left_yaw])

#     t_out += 1
#     t_in += 1

# # Save final episode
# if traj_dir is not None and len(pose_buffer) > 0:
#     pose_arr = np.array(pose_buffer, dtype=np.float32)
#     np.save(os.path.join(traj_dir, "pose_4fps.npy"), pose_arr)

# print("\n4FPS episode extraction complete.")
# print("Saved to:", OUT_ROOT)

import os
import shutil
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import random

# config
EPISODE_ROOT = "/home/jeanine/nwm/xvla_soft_fold"
OUT_DATASET = "xvla_soft_fold_12"
DATA_ROOT = f"/home/jeanine/nwm/data/{OUT_DATASET}"
SPLIT_ROOT = f"/home/jeanine/nwm/data_splits/{OUT_DATASET}"

WINDOW = 12
STRIDE = 2
TRAIN_RATIO = 0.7
SEED = 0

random.seed(SEED)

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(os.path.join(SPLIT_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(SPLIT_ROOT, "test"), exist_ok=True)

# collect episodes
episodes = sorted(glob(os.path.join(EPISODE_ROOT, "episode_*")))
print("Found episodes:", len(episodes))

traj_counter = 0
traj_names = []

# process each episode
for ep_dir in tqdm(episodes, desc="Building trajectories"):

    pose_path = os.path.join(ep_dir, "pose_4fps.npy")
    if not os.path.exists(pose_path):
        continue

    pose = np.load(pose_path)
    T = pose.shape[0]

    img_paths = sorted(
        glob(os.path.join(ep_dir, "*.jpg")),
        key=lambda x: int(os.path.basename(x).split(".")[0])
    )

    assert len(img_paths) == T, "Image/Pose mismatch!"

    for start in range(0, T - WINDOW + 1, STRIDE):

        end = start + WINDOW

        traj_name = f"traj_{traj_counter:06d}"
        traj_dir = os.path.join(DATA_ROOT, traj_name)
        os.makedirs(traj_dir, exist_ok=True)

        # copy images
        for i in range(WINDOW):
            shutil.copy(
                img_paths[start + i],
                os.path.join(traj_dir, f"{i}.jpg")
            )

        # create traj_data.pkl
        window_pose = pose[start:end]

        traj_data = {
            "position": window_pose[:, :2].astype(np.float32),
            "yaw": window_pose[:, 2].astype(np.float32),
        }

        with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)

        traj_names.append(traj_name)
        traj_counter += 1

print("\nTotal trajectories built:", traj_counter)

# create splits
random.shuffle(traj_names)

num_train = int(len(traj_names) * TRAIN_RATIO)

train_names = traj_names[:num_train]
test_names = traj_names[num_train:]

# write train split
with open(os.path.join(SPLIT_ROOT, "train", "traj_names.txt"), "w") as f:
    for name in train_names:
        f.write(name + "\n")

# write test split
with open(os.path.join(SPLIT_ROOT, "test", "traj_names.txt"), "w") as f:
    for name in test_names:
        f.write(name + "\n")

print("Train trajectories:", len(train_names))
print("Test trajectories:", len(test_names))
print("Saved splits to:", SPLIT_ROOT)
