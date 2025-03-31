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
import os
import transformer_engine.pytorch 
import imageio
import torch

from cosmos_predict1.autoregressive.inference.world_generation_pipeline import ARBaseGenerationPipeline
from cosmos_predict1.autoregressive.utils.inference import add_common_arguments, load_vision_input, validate_args, retrieve_raw_video_path, prepare_raw_video
from cosmos_predict1.utils import log

def parse_args():
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)
    parser.add_argument(
        "--ar_model_dir",
        type=str,
        default="Cosmos-Predict1-4B",
    )
    parser.add_argument("--enable_tokenizer", action="store_true", help="Enable the tokenizer")
    parser.add_argument("--input_type", type=str, default="video", help="Type of input", choices=["image", "video", "tokens"])
    parser.add_argument("--only_eval", action="store_true", help="calculate cross-entropy across entire dataset")

    args = parser.parse_args()
    return args


def main(args):
    """Run video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple images/videos from input
    - Generating videos from images/videos
    - Saving the generated videos to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (temperature, top_p)
            - Input/output settings (images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    inference_type = "base"  # When the inference_type is "base", AR model does not take text as input, the world generation is purely based on the input video
    sampling_config = validate_args(args, inference_type)

    if args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()

    # Initialize base generation model pipeline
    pipeline = ARBaseGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.ar_model_dir,
        disable_diffusion_decoder=args.disable_diffusion_decoder,
        offload_guardrail_models=args.offload_guardrail_models,
        offload_diffusion_decoder=args.offload_diffusion_decoder,
        offload_network=args.offload_ar_model,
        offload_tokenizer=args.offload_tokenizer,
        disable_guardrail=args.disable_guardrail,
        parallel_size=args.num_gpus,
    )

    # for single video, this is dict with just one key-value pair
    # should be dict[os.path.basename(video_path)] = video: torch.Tensor

    input_map = retrieve_raw_video_path(args.input_image_or_video_path)
    input_videos = prepare_raw_video(input_map, args.num_input_frames)

    # input_videos = load_vision_input(
    #     input_type=args.input_type,
    #     batch_input_path=args.batch_input_path,
    #     input_image_or_video_path=args.input_image_or_video_path,
    #     data_resolution=args.data_resolution,
    #     num_input_frames=args.num_input_frames,
    # )
    print("completed loading input videos")

    # @NOTE:
    # for single video, this is dict with just one key-value pair
    # should be dict[os.path.basename(video_path)] = video: torch.Tensor
    # the video contains num_input_frames last frames of the original video
    # then it contains NUM_TOTAL_FRAMES frames after padding which is just the last frame repeated NUM_TOTAL_FRAMES times
    # the video is of shape (1, NUM_TOTAL_FRAMES, 3, H, W)
    for idx, input_filename in enumerate(input_videos["video_prompt"]):
        # this is the video tensor of shape (1, NUM_TOTAL_FRAMES, 3, H, W)
        inp_vid = input_videos["video_prompt"][input_filename]
        gt_vid = input_videos["video_gt"][input_filename]
        # Generate video
        log.info(f"Run with image or video path: {input_filename}")
        out_vid = pipeline.generate(
            inp_vid=inp_vid,
            gt_vid=gt_vid,
            num_input_frames=args.num_input_frames,
            seed=args.seed,
            sampling_config=sampling_config,
            enable_tokenizer=args.enable_tokenizer,
        )
        if out_vid is None:
            log.critical("Guardrail blocked base generation.")
            continue

        # Create output directory if it doesn't exist
        out_dir = os.path.join(args.video_save_folder, "raw", "generated")
        os.makedirs(out_dir, exist_ok=True)

        # Save video
        if args.input_image_or_video_path:
            out_vid_path = os.path.join(out_dir, f"{input_filename}.mp4")
        else:
            out_vid_path = os.path.join(out_dir, f"{input_filename}.mp4")
        
        # Debug information about the returned tuple
        print(f"Type of out_vid: {type(out_vid)}")
        print(f"Length of out_vid tuple: {len(out_vid)}")
        for i, item in enumerate(out_vid):
            print(f"Item {i} type: {type(item)}")
            if isinstance(item, torch.Tensor):
                print(f"Item {i} shape: {item.shape}")
        
        # Process the video frames from the output tuple
        video_data = out_vid[0]  # First item in the tuple is the video data
        
        # Now handle differently depending on what type of data we have
        if isinstance(video_data, list) and len(video_data) == 1 and hasattr(video_data[0], 'shape') and len(video_data[0].shape) == 4:
            # If it's a list containing a single tensor with shape [frames, H, W, C]
            frames_tensor = video_data[0]
            out_vid_list = [frames_tensor[i].numpy() if isinstance(frames_tensor, torch.Tensor) 
                            else frames_tensor[i] for i in range(frames_tensor.shape[0])]
        elif isinstance(video_data, list):
            # If it's already a list of individual frames
            out_vid_list = [frame.numpy() if isinstance(frame, torch.Tensor) else frame 
                           for frame in video_data]
        else:
            # Fallback case
            out_vid_list = video_data
            
        print(f"Processed out_vid_list length: {len(out_vid_list)}")
        if len(out_vid_list) > 0:
            print(f"First frame shape: {out_vid_list[0].shape}")
        
        imageio.mimsave(out_vid_path, out_vid_list, fps=30)
        log.info(f"Saved video to {out_vid_path}")

    # clean up properly distributed
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    torch._C._jit_set_texpr_fuser_enabled(False)
    args = parse_args()
    main(args)
