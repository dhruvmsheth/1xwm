# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision
from PIL import Image

from cosmos_predict1.autoregressive.configs.inference.inference_config import SamplingConfig
from cosmos_predict1.utils import log

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]
_VIDEO_EXTENSIONS = [".mp4"]
_SUPPORTED_CONTEXT_LEN = [1, 9]  # Input frames
NUM_TOTAL_FRAMES = 17

# meaning of each input:
# pixel chunk duration represents the number of frames in each chunk of the video. 
# num total frames represents the total number of frames in the video
# num condition latents t means the number of frames in the video that are used as condition for the video generation
# video height and videos width represent the height and width of the video
INPUTS = {
    "PIXEL_CHUNK_DURATION_I": 17,
    "NUM_TOTAL_FRAMES_I": NUM_TOTAL_FRAMES,
    "NUM_CONDITION_LATENTS_T_I": 1,
    "VIDEO_HEIGHT_I": 512,
    "VIDEO_WIDTH_I": 512,
    "COMPRESSION_RATIO_I": [8, 16, 16],
}

def add_common_arguments(parser):
    """Add common command line arguments.

    Args:
        parser (ArgumentParser): Argument parser to add arguments to
    """
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--video_save_name",
        type=str,
        default="output",
        help="Output filename for generating a single video",
    )
    parser.add_argument("--video_save_folder", type=str, default="outputs/", help="Output folder for saving videos")
    parser.add_argument(
        "--input_image_or_video_path",
        type=str,
        help="Input path for input image or video",
    )
    parser.add_argument(
        "--input_tokens_dir",
        type=str,
        help="Input directory for input tokens",
    )
    parser.add_argument(
        "--batch_input_path",
        type=str,
        help="Input folder containing all input images or videos",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=9,
        help="Number of input frames for world generation",
        choices=_SUPPORTED_CONTEXT_LEN,
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p value for sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs used to run inference in parallel.")
    parser.add_argument("--disable_diffusion_decoder", action="store_true", help="Disable diffusion decoder")
    parser.add_argument(
        "--offload_guardrail_models",
        action="store_true",
        help="Offload guardrail models after inference",
    )
    parser.add_argument(
        "--offload_diffusion_decoder",
        action="store_true",
        help="Offload diffusion decoder after inference",
    )
    parser.add_argument(
        "--offload_ar_model",
        action="store_true",
        help="Offload AR model after inference",
    )
    parser.add_argument(
        "--offload_tokenizer",
        action="store_true",
        help="Offload discrete tokenizer model after inference",
    )
    parser.add_argument(
        "--disable_guardrail",
        action="store_true",
        help="Disable guardrail models",
    )


def validate_args(args: argparse.Namespace, inference_type: str):
    """Validate command line arguments for base and video2world generation."""
    assert inference_type in [
        "base",
        "video2world",
    ], "Invalid inference_type, must be 'base' or 'video2world'"
    if args.input_type in ["image", "text_and_image"] and args.num_input_frames != 1:
        args.num_input_frames = 1
        log.info(f"Set num_input_frames to 1 for {args.input_type} input")

    if args.num_input_frames == 1:
        if "4B" in args.ar_model_dir:
            log.warning(
                "The failure rate for 4B model with image input is ~15%. 12B / 13B model have a smaller failure rate. Please be cautious and refer to README.md for more details."
            )
        elif "5B" in args.ar_model_dir:
            log.warning(
                "The failure rate for 5B model with image input is ~7%. 12B / 13B model have a smaller failure rate. Please be cautious and refer to README.md for more details."
            )

    # Validate prompt/image/video args for single or batch generation
    assert (
        args.input_image_or_video_path or args.batch_input_path or args.input_tokens_dir
    ), "--input_image_or_video_path or --batch_input_path or --input_tokens_dir must be provided."
    if inference_type == "video2world" and (not args.batch_input_path):
        assert args.prompt, "--prompt is required for single video generation."
    args.data_resolution = [INPUTS["VIDEO_HEIGHT_I"], INPUTS["VIDEO_WIDTH_I"]]

    # Create output folder
    Path(args.video_save_folder).mkdir(parents=True, exist_ok=True)

    sampling_config = SamplingConfig(
        echo=True,
        temperature=args.temperature,
        top_p=args.top_p,
        compile_sampling=True,
    )
    return sampling_config

def retrieve_token_path(token_dir: str):
    token_path = {
        "tokens": os.path.join(token_dir, "video_0.bin"),
        "states": os.path.join(token_dir, "states_0.bin"),
        "metadata": os.path.join(token_dir, "metadata_0.json"),
        "segments": os.path.join(token_dir, "segments_idx_0.json"),
    }
    return token_path

def retrieve_raw_video_path(video_dir: str):
    video_path = {
        "videos": os.path.join(video_dir, "video_0.mp4"),
        "states": os.path.join(video_dir, "states_0.bin"),
        "metadata": os.path.join(video_dir, "metadata_0.json"),
        "segments": os.path.join(video_dir, "segments_idx_0.json"),
    }
    return video_path

def prepare_raw_video(input_map: dict[str, str], num_input_frames: int) -> dict[str, dict]:
    """
    Prepares video with lazy loading.
    
    Args:
        input_map (dict[str, str]): Dictionary containing input paths
        num_input_frames (int): Number of input frames to use
        
    Returns:
        dict: Dictionary containing video_gt and video_prompt dictionaries
    """
    video_path = input_map["videos"]
    metadata_path = input_map["metadata"]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    total_frames = metadata["shard_num_frames"]
    total_chunks = math.ceil(total_frames / 17)
    print(f"Total chunks in shard: {total_chunks}")
    
    # Create lazy video loaders
    video_gt_dict = LazyChunkDict(
        video_path=video_path,
        total_chunks=total_chunks,
        num_input_frames=num_input_frames,
        is_prompt=False
    )
    
    video_prompt_dict = LazyChunkDict(
        video_path=video_path,
        total_chunks=total_chunks,
        num_input_frames=num_input_frames,
        is_prompt=True
    )
    
    return {
        "video_gt": video_gt_dict,
        "video_prompt": video_prompt_dict
    }


class LazyChunkDict:
    """Dictionary-like object that lazily loads video chunks when accessed."""
    
    def __init__(self, video_path, total_chunks, num_input_frames, is_prompt):
        self.video_path = video_path
        self.total_chunks = total_chunks
        self.num_input_frames = num_input_frames
        self.is_prompt = is_prompt
        self.frames_per_chunk = NUM_TOTAL_FRAMES
        self.chunk_cache = {}  # Cache for loaded chunks
        self.video_data = None
    
    def __getitem__(self, key):
        """Lazily load a chunk when accessed by key."""
        # Initialize video data if not already done
        self._initialize_video()
        chunk_idx = int(key.split("_")[1])
        chunk = self.video_data[chunk_idx]
        if self.is_prompt:
            result = torch.zeros_like(chunk)
            result[:self.num_input_frames] = chunk[:self.num_input_frames]
            last_frame = chunk[self.num_input_frames-1:self.num_input_frames]
            result[self.num_input_frames:] = last_frame.repeat(
                self.frames_per_chunk - self.num_input_frames, 1, 1, 1
            )
        else:
            result = chunk
        # Add batch dimension to match expected shape (1, NUM_TOTAL_FRAMES, 3, H, W)
        result = result.unsqueeze(0)
        return result
    
    def _initialize_video(self):
        """Lazy initialize video data."""
        if self.video_data is None:
            print(f"Initializing video data for {'prompt' if self.is_prompt else 'ground truth'} (first access)")
            
            # Read the entire video
            video, _, _ = torchvision.io.read_video(self.video_path, pts_unit="sec")
            video = video.float() / 255.0 
            video = video * 2 - 1
            video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
            total_needed = self.total_chunks * self.frames_per_chunk
            if video.shape[0] < total_needed:
                pad_frames = total_needed - video.shape[0]
                video = torch.cat([video, video[-1:].repeat(pad_frames, 1, 1, 1)], dim=0)
            self.video_data = video.reshape(self.total_chunks, self.frames_per_chunk, 3, 512, 512)
    
    def __iter__(self):
        """Allow iteration over keys."""
        return iter([f"sample_{i}" for i in range(self.total_chunks)])
    
    def keys(self):
        """Return all possible keys."""
        return [f"sample_{i}" for i in range(self.total_chunks)]
    
    def __len__(self):
        """Return number of chunks."""
        return self.total_chunks


def resize_input(video: torch.Tensor, resolution: list[int]):
    r"""
    Function to perform aspect ratio preserving resizing and center cropping.
    This is needed to make the video into target resolution.
    Args:
        video (torch.Tensor): Input video tensor
        resolution (list[int]): Data resolution
    Returns:
        Cropped video
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def load_image_from_list(flist, data_resolution: List[int]) -> dict:
    """
    Function to load images from a list of image paths.
    Args:
        flist (List[str]): List of image paths
        data_resolution (List[int]): Data resolution
    Returns:
        Dict containing input images
    """
    all_videos = dict()
    for img_path in flist:
        ext = os.path.splitext(img_path)[1]
        if ext in _IMAGE_EXTENSIONS:
            # Read the image
            img = Image.open(img_path)

            # Convert to tensor
            img = torchvision.transforms.functional.to_tensor(img)
            static_vid = img.unsqueeze(0).repeat(NUM_TOTAL_FRAMES, 1, 1, 1)
            static_vid = static_vid * 2 - 1

            log.debug(
                f"Resizing input image of shape ({static_vid.shape[2]}, {static_vid.shape[3]}) -> ({data_resolution[0]}, {data_resolution[1]})"
            )
            static_vid = resize_input(static_vid, data_resolution)
            fname = os.path.basename(img_path)
            all_videos[fname] = static_vid.transpose(0, 1).unsqueeze(0)

    return all_videos


def read_input_images(batch_input_path: str, data_resolution: List[int]) -> dict:
    """
    Function to read input images from a JSONL file.

    Args:
        batch_input_path (str): Path to JSONL file containing visual input paths
        data_resolution (list[int]): Data resolution

    Returns:
        Dict containing input images
    """
    # Read visual inputs from JSONL
    flist = []
    with open(batch_input_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            flist.append(data["visual_input"])

    return load_image_from_list(flist, data_resolution=data_resolution)


def read_input_image(input_path: str, data_resolution: List[int]) -> dict:
    """
    Function to read input image.
    Args:
        input_path (str): Path to input image
        data_resolution (List[int]): Data resolution
    Returns:
        Dict containing input image
    """
    flist = [input_path]
    return load_image_from_list(flist, data_resolution=data_resolution)


def read_input_videos(batch_input_path: str, data_resolution: List[int], num_input_frames: int) -> dict:
    r"""
    Function to read input videos.
    Args:
        batch_input_path (str): Path to JSONL file containing visual input paths
        data_resolution (list[int]): Data resolution
    Returns:
        Dict containing input videos
    """
    # Read visual inputs from JSONL
    flist = []
    with open(batch_input_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            flist.append(data["visual_input"])
    # flist is None here since for single video, we don't need to read from JSONL
    return load_videos_from_list(flist, data_resolution=data_resolution, num_input_frames=num_input_frames)


def read_input_video(input_path: str, data_resolution: List[int], num_input_frames: int) -> dict:
    """
    Function to read input video.
    Args:
        input_path (str): Path to input video
        data_resolution (List[int]): Data resolution
        num_input_frames (int): Number of frames in context
    Returns:
        Dict containing input video
    """
    flist = [input_path]
    return load_videos_from_list(flist, data_resolution=data_resolution, num_input_frames=num_input_frames)


def load_videos_from_list(flist: List[str], data_resolution: List[int], num_input_frames: int) -> dict:
    """
    Function to load videos from a list of video paths.
    Args:
        flist (List[str]): List of video paths
        data_resolution (List[int]): Data resolution
        num_input_frames (int): Number of frames in context
    Returns:
        Dict containing input videos
    """
    # flist is None here since for single video, we don't need to read from JSONL
    all_videos = dict()

    for video_path in flist:
        # only single video for our case here
        ext = os.path.splitext(video_path)[-1]
        if ext in _VIDEO_EXTENSIONS:
            video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
            video = video.float() / 255.0
            video = video * 2 - 1

            # Resize the videos to the required dimension
            nframes_in_video = video.shape[0]
            if nframes_in_video < num_input_frames:
                fname = os.path.basename(video_path)
                log.warning(
                    f"Video {fname} has {nframes_in_video} frames, less than the requried {num_input_frames} frames. Skipping."
                )
                continue
            # only keep the last num_input_frames frames
            video = video[-num_input_frames:, :, :, :]

            # Pad the video to NUM_TOTAL_FRAMES (because the tokenizer expects inputs of NUM_TOTAL_FRAMES)
            video = torch.cat(
                (video, video[-1, :, :, :].unsqueeze(0).repeat(NUM_TOTAL_FRAMES - num_input_frames, 1, 1, 1)),
                dim=0,
            )
            # video is of shape (NUM_TOTAL_FRAMES, 3, 512, 512)
            video = video.permute(0, 3, 1, 2)

            log.debug(
                f"Resizing input video of shape ({video.shape[2]}, {video.shape[3]}) -> ({data_resolution[0]}, {data_resolution[1]})"
            )
            video = resize_input(video, data_resolution)

            fname = os.path.basename(video_path)
            all_videos[fname] = video.transpose(0, 1).unsqueeze(0)
    # all videos here is dict with just one key-value pair
    return all_videos

def load_vision_input(
    input_type: str,
    batch_input_path: str,
    input_image_or_video_path: str,
    data_resolution: List[int],
    num_input_frames: int,
) -> dict[str, torch.Tensor]:
    """
    Function to load vision input.
    Note: We pad the frames of the input image/video to NUM_TOTAL_FRAMES here, and feed the padded video tensors to the video tokenizer to obtain tokens. The tokens will be truncated based on num_input_frames when feeding to the autoregressive model.
    Args:
        input_type (str): Type of input
        batch_input_path (str): Folder containing input images or videos
        input_image_or_video_path (str): Path to input image or video
        data_resolution (List[int]): Data resolution
        num_input_frames (int): Number of frames in context
    Returns:
        Dict containing input videos
    """
    # @NOTE:
    # for single video, we don't need to read from JSONL
    if batch_input_path:
        log.info(f"Reading batch inputs from path: {batch_input_path}")
        if input_type == "image" or input_type == "text_and_image":
            input_videos = read_input_images(batch_input_path, data_resolution=data_resolution)
        # 1x challenge will be using this
        elif input_type == "video" or input_type == "text_and_video":

            input_videos = read_input_videos(
                batch_input_path,
                data_resolution=data_resolution,
                num_input_frames=num_input_frames,
            )
        else:
            raise ValueError(f"Invalid input type {input_type}")
    # @NOTE:
    else:
        if input_type == "image" or input_type == "text_and_image":
            input_videos = read_input_image(input_image_or_video_path, data_resolution=data_resolution)
        # 1x challenge will be using this
        # @NOTE:
        elif input_type == "video" or input_type == "text_and_video":
            # this is dict with just one key-value pair
            # should be dict[os.path.basename(video_path)] = video: torch.Tensor
            input_videos = read_input_video(
                input_image_or_video_path,
                data_resolution=data_resolution,
                num_input_frames=num_input_frames,
            )
        else:
            raise ValueError(f"Invalid input type {input_type}")
    return input_videos


def prepare_video_batch_for_saving(video_batch: List[torch.Tensor]) -> List[np.ndarray]:
    """
    Function to convert output tensors to numpy format for saving.
    Args:
        video_batch (List[torch.Tensor]): List of output tensors
    Returns:
        List of numpy arrays
    """
    return [(video * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy() for video in video_batch]
