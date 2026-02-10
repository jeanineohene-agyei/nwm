# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# from lerobot.datasets.lerobot_dataset import LeRobotDataset


# # ======================
# # CONFIG
# # ======================
# REPO_ID = "lerobot/xvla-soft-fold"
# OUT_ROOT = "/home/jeanine/nwm/soft_fold_positions"

# CAMERA_KEY = "cam_high"
# OUT_W, OUT_H = 320, 240

# ORIG_FPS = 50
# TARGET_FPS = 4
# STRIDE = 12
# DT = 1.0 / TARGET_FPS

# MAX_EPISODES = 12

# os.makedirs(OUT_ROOT, exist_ok=True)


# # ======================
# # LOAD DATASET
# # ======================
# print("Loading LeRobot dataset...")
# ds = LeRobotDataset(REPO_ID)
# print("Dataset loaded.")
# print("Total samples:", len(ds))


# # ======================
# # EXPORT
# # ======================
# current_episode = None
# episodes_seen = 0

# t_in = 0
# t_out = 0

# state_buffer = []
# traj_dir = None

# for sample in tqdm(ds, desc="Exporting episodes"):

#     ep_id = int(sample["episode_index"])

#     # stop after first N episodes
#     if current_episode is not None and ep_id != current_episode:
#         episodes_seen += 1
#         if episodes_seen >= MAX_EPISODES:
#             break

#     # ------------------
#     # new episode
#     # ------------------
#     if current_episode is None or ep_id != current_episode:

#         # save previous episode state
#         if traj_dir is not None and len(state_buffer) > 0:
#             state_arr = np.stack(state_buffer, axis=0).astype(np.float32)
#             np.save(os.path.join(traj_dir, "state.npy"), state_arr)

#         current_episode = ep_id
#         traj_dir = os.path.join(OUT_ROOT, f"episode_{ep_id:06d}")
#         os.makedirs(traj_dir, exist_ok=True)

#         t_in = 0
#         t_out = 0
#         state_buffer = []

#     # ------------------
#     # temporal downsampling
#     # ------------------
#     if t_in % STRIDE != 0:
#         t_in += 1
#         continue

#     # ------------------
#     # IMAGE PROCESSING
#     # ------------------
#     img = sample[f"observation.images.{CAMERA_KEY}"]  # torch [3,H,W]

#     img = img.permute(1, 2, 0).cpu().numpy()
#     img = np.clip(img, 0.0, 1.0)
#     img = (img * 255.0).astype(np.uint8)

#     img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
#     img = img[:, :, ::-1]  # RGB → BGR

#     cv2.imwrite(os.path.join(traj_dir, f"{t_out}.jpg"), img)

#     # ------------------
#     # STATE PROCESSING
#     # ------------------
#     state = sample["observation.state"].cpu().numpy()  # (96,)
#     state_buffer.append(state)

#     t_out += 1
#     t_in += 1


# # ======================
# # SAVE FINAL EPISODE
# # ======================
# if traj_dir is not None and len(state_buffer) > 0:
#     state_arr = np.stack(state_buffer, axis=0).astype(np.float32)
#     np.save(os.path.join(traj_dir, "state.npy"), state_arr)

# print("Export complete.")
# print("Episodes exported:", MAX_EPISODES)
# print("Output directory:", OUT_ROOT)


# import os
# import shutil
# import pickle
# import numpy as np
# from glob import glob
# from tqdm import tqdm
# import random

# # ======================
# # CONFIG
# # ======================
# SRC_EPISODES = "/home/jeanine/nwm/aloha_static_cups_open"
# OUT_DATASET = "aloha_static_cups_open"
# OUT_ROOT = f"nwm/data/{OUT_DATASET}"

# CONTEXT_LEN = 4
# PRED_LEN = 8
# WINDOW = CONTEXT_LEN + PRED_LEN
# STRIDE = 2

# NUM_TEST_TRAJS = 10
# SEED = 0

# random.seed(SEED)

# os.makedirs(OUT_ROOT, exist_ok=True)

# # ======================
# # COLLECT EPISODES
# # ======================
# episodes = sorted(glob(os.path.join(SRC_EPISODES, "episode_*")))
# print(f"Found {len(episodes)} episodes")

# traj_names = []

# traj_counter = 0

# # ======================
# # PROCESS EPISODES
# # ======================
# for ep_idx, ep_dir in enumerate(tqdm(episodes, desc="Processing episodes")):
#     frames = sorted(glob(os.path.join(ep_dir, "*.jpg")))
#     num_frames = len(frames)

#     if num_frames < WINDOW:
#         continue

#     start = 0
#     traj_in_ep = 0

#     while start + WINDOW <= num_frames:
#         ep_name = os.path.basename(ep_dir)   # "episode_000013"
#         traj_name = f"{ep_name}_s{start:04d}"

#         traj_dir = os.path.join(OUT_ROOT, traj_name)
#         os.makedirs(traj_dir, exist_ok=True)

#         # ------------------
#         # COPY FRAMES
#         # ------------------
#         for i in range(WINDOW):
#             src = frames[start + i]
#             dst = os.path.join(traj_dir, f"{i}.jpg")
#             shutil.copy(src, dst)

#         # ------------------
#         # DUMMY TRAJ DATA
#         # ------------------
#         traj_data = {
#         "position": np.zeros((WINDOW, 2), dtype=np.float32),
#         "yaw": np.zeros((WINDOW,), dtype=np.float32)
#     }

#         with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
#             pickle.dump(traj_data, f)

#         traj_names.append(traj_name)

#         traj_counter += 1
#         traj_in_ep += 1
#         start += STRIDE

# print(f"Total trajectories created: {traj_counter}")

# # ======================
# # TRAIN / TEST SPLIT
# # ======================
# random.shuffle(traj_names)

# test_trajs = traj_names[:NUM_TEST_TRAJS]
# train_trajs = traj_names[NUM_TEST_TRAJS:]

# print(f"Train trajs: {len(train_trajs)}")
# print(f"Test trajs: {len(test_trajs)}")

# # ======================
# # WRITE SPLITS
# # ======================
# split_root = f"/home/jeanine/nwm/data_splits/{OUT_DATASET}"

# train_split = os.path.join(split_root, "train")
# test_split = os.path.join(split_root, "test")

# os.makedirs(train_split, exist_ok=True)
# os.makedirs(test_split, exist_ok=True)

# with open(os.path.join(train_split, "traj_names.txt"), "w") as f:
#     for name in train_trajs:
#         f.write(f"{name}\n")

# with open(os.path.join(test_split, "traj_names.txt"), "w") as f:
#     for name in test_trajs:
#         f.write(f"{name}\n")

import os
import shutil
import pickle
import numpy as np
from glob import glob

SRC_ROOT = "/home/jeanine/nwm/soft_fold_positions"
OUT_ROOT = "/home/jeanine/nwm/data/soft_fold_positions"
SPLIT_ROOT = "/home/jeanine/nwm/data_splits/soft_fold_positions"

WINDOW = 12
STRIDE = 2

TRAIN_SPLIT = os.path.join(SPLIT_ROOT, "train")
TEST_SPLIT = os.path.join(SPLIT_ROOT, "test")

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(TRAIN_SPLIT, exist_ok=True)
os.makedirs(TEST_SPLIT, exist_ok=True)

train_traj_names = []
test_traj_names = []

episodes = sorted(glob(os.path.join(SRC_ROOT, "episode_*")))
assert len(episodes) > 0, "No episodes found"

last_episode = episodes[-1]

for ep_idx, ep_dir in enumerate(episodes):
    ep_name = os.path.basename(ep_dir)

    frames = sorted(glob(os.path.join(ep_dir, "*.jpg")))
    num_frames = len(frames)

    # 🔹 LOAD STATE (THIS IS THE KEY CHANGE)
    state_path = os.path.join(ep_dir, "state.npy")
    assert os.path.exists(state_path), f"Missing state.npy in {ep_dir}"

    states = np.load(state_path)  # shape (T, 96)
    assert states.shape[0] == num_frames, "State/image length mismatch"

    start = 0
    while start + WINDOW <= num_frames:
        traj_name = f"{ep_name}_s{start:03d}"
        traj_dir = os.path.join(OUT_ROOT, traj_name)
        os.makedirs(traj_dir, exist_ok=True)

        # ------------------
        # copy frames
        # ------------------
        for i in range(WINDOW):
            src = frames[start + i]
            dst = os.path.join(traj_dir, f"{i}.jpg")
            shutil.copy(src, dst)

        # ------------------
        # build traj_data.pkl (REAL POSE)
        # ------------------
        pos = []
        yaw = []

        for t in range(start, start + WINDOW):
            state = states[t]

            # position from eef6d
            x = state[30]
            y = state[31]

            # yaw from eef_euler_2
            theta = state[2]

            pos.append([x, y])
            yaw.append(theta)

        traj_data = {
            "position": np.array(pos, dtype=np.float32),
            "yaw": np.array(yaw, dtype=np.float32),
        }

        with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)

        # ------------------
        # assign split
        # ------------------
        if ep_dir == last_episode:
            test_traj_names.append(traj_name)
        else:
            train_traj_names.append(traj_name)

        start += STRIDE

# ------------------
# write splits
# ------------------
with open(os.path.join(TRAIN_SPLIT, "traj_names.txt"), "w") as f:
    for name in train_traj_names:
        f.write(f"{name}\n")

with open(os.path.join(TEST_SPLIT, "traj_names.txt"), "w") as f:
    for name in test_traj_names:
        f.write(f"{name}\n")

print(f"Train trajectories: {len(train_traj_names)}")
print(f"Test trajectories: {len(test_traj_names)}")
print("Done.")
