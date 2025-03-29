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
from cosmos_predict1.autoregressive.utils.inference import add_common_arguments, load_vision_input, validate_args, retrieve_token_path
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
        enable_tokenizer=args.enable_tokenizer,
    )

    # for single video, this is dict with just one key-value pair
    # should be dict[os.path.basename(video_path)] = video: torch.Tensor
    if args.input_type == "video" or args.input_type == "image":
        input_videos = load_vision_input(
            input_type=args.input_type,
            batch_input_path=args.batch_input_path,
            input_image_or_video_path=args.input_image_or_video_path,
            data_resolution=args.data_resolution,
            num_input_frames=args.num_input_frames,
        )
    else:
        input_videos = None
    print("completed loading input videos")

    token_map = retrieve_token_path(args.input_tokens_dir)

    # @NOTE:
    # for single video, this is dict with just one key-value pair
    # should be dict[os.path.basename(video_path)] = video: torch.Tensor
    # the video contains num_input_frames last frames of the original video
    # then it contains NUM_TOTAL_FRAMES frames after padding which is just the last frame repeated NUM_TOTAL_FRAMES times
    # the video is of shape (1, NUM_TOTAL_FRAMES, 3, H, W)
        # this is the video tensor of shape (1, NUM_TOTAL_FRAMES, 3, H, W)
        # Generate video
    log.info(f"Run with image or video path: {args.input_tokens_dir}")
    data_tokens, token_boundaries = pipeline.load_tokens(token_map)
    print(f"data_tokens shape: {data_tokens.shape}")

    for i in range(data_tokens.shape[0]):
        print("Processing sample: ", i)
        data_token_cur_batch = data_tokens[i]
        token_boundary_cur_batch = token_boundaries["video"][i]

        out_vid, ce_loss = pipeline.generate(
            inp_vid=input_videos,
            num_input_frames=args.num_input_frames,
            seed=args.seed,
            sampling_config=sampling_config,
            data_token=data_token_cur_batch,
            token_boundary=token_boundary_cur_batch,
            only_eval=args.only_eval
        )
        if out_vid is None and not args.only_eval:
            log.critical("Guardrail blocked base generation.")
        elif out_vid is None and args.only_eval:
            log.info("Done with evaluation!")

        # Save video
        # mkdir generated_videos and generated_loss
        os.makedirs(args.video_save_folder, exist_ok=True)
        os.makedirs(os.path.join(args.video_save_folder, "generated_videos"), exist_ok=True)
        os.makedirs(os.path.join(args.video_save_folder, "generated_loss"), exist_ok=True)
        if args.input_image_or_video_path:
            out_vid_path = os.path.join(args.video_save_folder, f"sample_{i}.mp4")
            out_ce_loss_path = os.path.join(args.video_save_folder, f"sample_{i}_ce_loss.txt")
        else:
            out_vid_path = os.path.join(args.video_save_folder, "generated_videos", f"sample_{i}_image.mp4")
            out_ce_loss_path = os.path.join(args.video_save_folder, "generated_loss", f"sample_{i}_ce_loss_image.txt")
        # expected out_vid shape: (1, 17, 256, 256, 3)
        # Convert out_vid to list of numpy arrays
        out_vid_list = [frame for frame in out_vid[0]]
        imageio.mimsave(out_vid_path, out_vid_list, fps=30)
        log.info(f"Saved video to {out_vid_path}")

        with open(out_ce_loss_path, "w") as f:
            f.write(f"CE loss: {ce_loss}")
        print(f"Saved CE loss {ce_loss} to {out_ce_loss_path}")
        print("--------------------------------")
        print("--------------------------------")

        # clean up properly distributed
        if args.num_gpus > 1:
            parallel_state.destroy_model_parallel()
            import torch.distributed as dist

            dist.destroy_process_group()


if __name__ == "__main__":
    torch._C._jit_set_texpr_fuser_enabled(False)
    args = parse_args()
    main(args)
