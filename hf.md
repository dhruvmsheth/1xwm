---
license: apache-2.0
pretty_name: 1X World Model Challenge Dataset
size_categories:
- 10M<n<100M
viewer: false
---
Dataset for the [1X World Model Challenge](https://github.com/1x-technologies/1xgpt).

Download with:
```
huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data
```

Changes from v1.1:
- New train and val dataset of 100 hours, replacing the v1.1 datasets
- Blur applied to faces
- Shared a new raw video dataset under CC-BY-NC-SA 4.0: https://huggingface.co/datasets/1x-technologies/worldmodel_raw_data
- Example scripts to decode Cosmos Tokenized bins `cosmos_video_decoder.py` and load in frame data `unpack_data.py`

Contents of train/val_v2.0:

The training dataset is shareded into 100 independent shards. The definitions are as follows:

- **video_{shard}.bin**: 8x8x8 image patches at 30hz, with 17 frame temporal window, encoded using [NVIDIA Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) "Cosmos-Tokenizer-DV8x8x8".
- **segment_idx_{shard}.bin** - Maps each frame `i` to its corresponding segment index. You may want to use this to separate non-contiguous frames from different videos (transitions).
- **states_{shard}.bin** - States arrays (defined below in `Index-to-State Mapping`) stored in `np.float32` format. For frame `i`, the corresponding state is represented by `states_{shard}[i]`.
- **metadata** - The `metadata.json` file provides high-level information about the entire dataset, while `metadata_{shard}.json` files contain specific details for each shard. 

  #### Index-to-State Mapping 
  ```
   {
        0: HIP_YAW
        1: HIP_ROLL
        2: HIP_PITCH
        3: KNEE_PITCH
        4: ANKLE_ROLL
        5: ANKLE_PITCH
        6: LEFT_SHOULDER_PITCH
        7: LEFT_SHOULDER_ROLL
        8: LEFT_SHOULDER_YAW
        9: LEFT_ELBOW_PITCH
        10: LEFT_ELBOW_YAW
        11: LEFT_WRIST_PITCH
        12: LEFT_WRIST_ROLL
        13: RIGHT_SHOULDER_PITCH
        14: RIGHT_SHOULDER_ROLL
        15: RIGHT_SHOULDER_YAW
        16: RIGHT_ELBOW_PITCH
        17: RIGHT_ELBOW_YAW
        18: RIGHT_WRIST_PITCH
        19: RIGHT_WRIST_ROLL
        20: NECK_PITCH
        21: Left hand closure state (0 = open, 1 = closed)
        22: Right hand closure state (0 = open, 1 = closed)
        23: Linear Velocity
        24: Angular Velocity
    }


Previous version: v1.1

- **magvit2.ckpt** - weights for [MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2) image tokenizer we used. We provide the encoder (tokenizer) and decoder (de-tokenizer) weights.

Contents of train/val_v1.1:
- **video.bin** - 16x16 image patches at 30hz, each patch is vector-quantized into 2^18 possible integer values. These can be decoded into 256x256 RGB images using the provided `magvig2.ckpt` weights.
- **segment_ids.bin** - for each frame `segment_ids[i]` uniquely points to the segment index that frame `i` came from. You may want to use this to separate non-contiguous frames from different videos (transitions).
- **actions/** - a folder of action arrays stored in `np.float32` format. For frame `i`, the corresponding action is given by `joint_pos[i]`, `driving_command[i]`, `neck_desired[i]`, and so on. The shapes and definitions of the arrays are as follows (N is the number of frames):
  - **joint_pos** `(N, 21)`: Joint positions. See `Index-to-Joint Mapping` below.  
  - **driving_command** `(N, 2)`: Linear and angular velocities.
  - **neck_desired** `(N, 1)`: Desired neck pitch.
  - **l_hand_closure** `(N, 1)`: Left hand closure state (0 = open, 1 = closed).
  - **r_hand_closure** `(N, 1)`: Right hand closure state (0 = open, 1 = closed).
  #### Index-to-Joint Mapping (OLD)
  ```
   {
        0: HIP_YAW
        1: HIP_ROLL
        2: HIP_PITCH
        3: KNEE_PITCH
        4: ANKLE_ROLL
        5: ANKLE_PITCH
        6: LEFT_SHOULDER_PITCH
        7: LEFT_SHOULDER_ROLL
        8: LEFT_SHOULDER_YAW
        9: LEFT_ELBOW_PITCH
        10: LEFT_ELBOW_YAW
        11: LEFT_WRIST_PITCH
        12: LEFT_WRIST_ROLL
        13: RIGHT_SHOULDER_PITCH
        14: RIGHT_SHOULDER_ROLL
        15: RIGHT_SHOULDER_YAW
        16: RIGHT_ELBOW_PITCH
        17: RIGHT_ELBOW_YAW
        18: RIGHT_WRIST_PITCH
        19: RIGHT_WRIST_ROLL
        20: NECK_PITCH
    }
  
  ```

  

We also provide a small `val_v1.1` data split containing held-out examples not seen in the training set, in case you want to try evaluating your model on held-out frames.