"""Reconstruct a video using Wan models.

This script encodes an input video with the Wan VAE and feeds the latent as
conditioning to the diffusion model. The base diffusion model can be selected
from the Wan2.1 text-to-video (T2V), image-to-video (I2V) or VACE pipelines.
"""

import argparse
import logging
import math

import torch
import torch.nn.functional as F
import torchvision.io as io

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.fm_solvers import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video


def load_video_tensor(path: str, frame_num: int, size: tuple[int, int],
                      device: torch.device) -> torch.Tensor:
    """Load a video, resample frames and resize to target size.

    Returns tensor with shape (3, F, H, W) in [-1,1].
    """

    video, _, _ = io.read_video(path)
    total = video.shape[0]
    if total != frame_num:
        idx = torch.linspace(0, total - 1, frame_num).long()
        video = video[idx]
    video = video.permute(0, 3, 1, 2).float() / 127.5 - 1.0
    video = F.interpolate(video, size=size, mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3).to(device)


def invert_with_base(pipe, video_latent, args, cfg):
    """Run diffusion inversion using WanT2V or WanI2V pipelines."""

    F = args.frame_num
    target_shape = video_latent.shape
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3]) /
        (pipe.patch_size[1] * pipe.patch_size[2]) * target_shape[1] /
        pipe.sp_size) * pipe.sp_size

    seed = args.seed if args.seed >= 0 else torch.randint(0, 2**31 - 1, (1,)).item()
    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)

    noise = [torch.randn_like(video_latent, generator=seed_g)]

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([""], pipe.device)
        context_null = pipe.text_encoder([""], pipe.device)
        pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([""], torch.device("cpu"))
        context_null = pipe.text_encoder([""], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]

    arg_c = {"context": context, "seq_len": seq_len, "y": [video_latent]}
    arg_null = {"context": context_null, "seq_len": seq_len, "y": [video_latent]}

    if isinstance(pipe, wan.WanI2V):
        img = pipe.vae.decode([video_latent])[0][:, 0]
        pipe.clip.model.to(pipe.device)
        clip_context = pipe.clip.visual([img[:, None, :, :]])
        pipe.clip.model.cpu()
        arg_c["clip_fea"] = clip_context
        arg_null["clip_fea"] = clip_context

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=pipe.param_dtype):
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipe.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(50, device=pipe.device, shift=5.0)
        timesteps = sample_scheduler.timesteps

        latents = noise
        for t in timesteps:
            latent_model_input = latents
            timestep = torch.tensor([t], device=pipe.device)
            pipe.model.to(pipe.device)
            noise_pred_cond = pipe.model(latent_model_input, t=timestep,
                                         **arg_c)[0]
            noise_pred_uncond = pipe.model(latent_model_input, t=timestep,
                                           **arg_null)[0]
            noise_pred = noise_pred_uncond + 0.0 * (noise_pred_cond -
                                                    noise_pred_uncond)
            temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t,
                                            latents[0].unsqueeze(0),
                                            return_dict=False,
                                            generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        pipe.model.cpu()
        torch.cuda.empty_cache()
        video = pipe.vae.decode(x0)[0]
    return video


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a video using Wan VAE conditioning")
    parser.add_argument("--ckpt_dir", required=True,
                        help="Path to model checkpoints")
    parser.add_argument("--src_video", required=True,
                        help="Path to the input video")
    parser.add_argument("--task",
                        default="t2v-14B",
                        choices=list(WAN_CONFIGS.keys()),
                        help="Model variant to use")
    parser.add_argument("--size",
                        default="1280*720",
                        choices=list(SIZE_CONFIGS.keys()),
                        help="Resolution of the generated video")
    parser.add_argument("--frame_num",
                        type=int,
                        default=81,
                        help="Number of frames to sample (4n+1)")
    parser.add_argument("--seed",
                        type=int,
                        default=-1,
                        help="Random seed (-1 for random)")
    parser.add_argument("--save_file",
                        default="reconstruction.mp4",
                        help="Where to save the reconstructed video")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = WAN_CONFIGS[args.task]
    device_id = 0

    size = SIZE_CONFIGS[args.size]

    if "vace" in args.task:
        wan_vace = wan.WanVace(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=0,
        )
        src_video, src_mask, src_ref_images = wan_vace.prepare_source(
            [args.src_video], [None], [None], args.frame_num, size,
            wan_vace.device)
        logging.info("Running diffusion with blank prompt")
        video = wan_vace.generate(
            "",
            src_video,
            src_mask,
            src_ref_images,
            size=size,
            frame_num=args.frame_num,
            context_scale=1.0,
            guide_scale=0.0,
            n_prompt="",
            seed=args.seed,
            offload_model=True,
        )
    else:
        pipe_cls = wan.WanI2V if "i2v" in args.task else wan.WanT2V
        pipe = pipe_cls(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=0,
        )
        video_tensor = load_video_tensor(args.src_video, args.frame_num, size,
                                         pipe.device)
        video_latent = pipe.vae.encode([video_tensor])[0]
        logging.info("Running diffusion with blank prompt")
        video = invert_with_base(pipe, video_latent, args, cfg)

    cache_video(video[None], save_file=args.save_file, fps=cfg.sample_fps, nrow=1)
    logging.info("Saved reconstructed video to %s", args.save_file)


if __name__ == "__main__":
    main()

