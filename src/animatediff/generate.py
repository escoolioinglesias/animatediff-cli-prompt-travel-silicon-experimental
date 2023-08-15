import logging
import re
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from diffusers import (AutoencoderKL, ControlNetModel, DiffusionPipeline,
                       StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionPipeline)
from tqdm.rich import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from animatediff import get_dir
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.schedulers import get_scheduler
from animatediff.settings import InferenceConfig, ModelConfig
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatediff.utils.model import (ensure_motion_modules,
                                     get_checkpoint_weights)
from animatediff.utils.util import get_resized_images, save_video

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
default_base_path = data_dir.joinpath("models/huggingface/stable-diffusion-v1-5")

re_clean_prompt = re.compile(r"[^\w\-, ]")


def load_safetensors_lora(text_encoder, unet, lora_path, alpha=0.75, is_animatediff=True):
    from safetensors.torch import load_file

    from animatediff.utils.lora_diffusers import (LoRANetwork,
                                                  create_network_from_weights)

    sd = load_file(lora_path)

    print(f"create LoRA network")
    lora_network: LoRANetwork = create_network_from_weights(text_encoder, unet, sd, multiplier=alpha, is_animatediff=is_animatediff)
    print(f"load LoRA network weights")
    lora_network.load_state_dict(sd, False)
    lora_network.merge_to(alpha)


def create_pipeline(
    base_model: Union[str, PathLike] = default_base_path,
    model_config: ModelConfig = ...,
    infer_config: InferenceConfig = ...,
    use_xformers: bool = True,
) -> AnimationPipeline:
    """Create an AnimationPipeline from a pretrained model.
    Uses the base_model argument to load or download the pretrained reference pipeline model."""

    # make sure motion_module is a Path and exists
    logger.info("Checking motion module...")
    motion_module = data_dir.joinpath(model_config.motion_module)
    if not (motion_module.exists() and motion_module.is_file()):
        # check for safetensors version
        motion_module = motion_module.with_suffix(".safetensors")
        if not (motion_module.exists() and motion_module.is_file()):
            # download from HuggingFace Hub if not found
            ensure_motion_modules()
        if not (motion_module.exists() and motion_module.is_file()):
            # this should never happen, but just in case...
            raise FileNotFoundError(f"Motion module {motion_module} does not exist or is not a file!")

    logger.info("Loading tokenizer...")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    logger.info("Loading text encoder...")
    text_encoder: CLIPSkipTextModel = CLIPSkipTextModel.from_pretrained(base_model, subfolder="text_encoder")
    logger.info("Loading VAE...")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    logger.info("Loading UNet...")
    unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path=base_model,
        motion_module_path=motion_module,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")

    # set up scheduler
    sched_kwargs = infer_config.noise_scheduler_kwargs
    scheduler = get_scheduler(model_config.scheduler, sched_kwargs)
    logger.info(f'Using scheduler "{model_config.scheduler}" ({scheduler.__class__.__name__})')

    # Load the checkpoint weights into the pipeline
    if model_config.path is not None:
        model_path = data_dir.joinpath(model_config.path)
        logger.info(f"Loading weights from {model_path}")
        if model_path.is_file():
            logger.debug("Loading from single checkpoint file")
            unet_state_dict, tenc_state_dict, vae_state_dict = get_checkpoint_weights(model_path)
        elif model_path.is_dir():
            logger.debug("Loading from Diffusers model directory")
            temp_pipeline = StableDiffusionPipeline.from_pretrained(model_path)
            unet_state_dict, tenc_state_dict, vae_state_dict = (
                temp_pipeline.unet.state_dict(),
                temp_pipeline.text_encoder.state_dict(),
                temp_pipeline.vae.state_dict(),
            )
            del temp_pipeline
        else:
            raise FileNotFoundError(f"model_path {model_path} is not a file or directory")

        # Load into the unet, TE, and VAE
        logger.info("Merging weights into UNet...")
        _, unet_unex = unet.load_state_dict(unet_state_dict, strict=False)
        if len(unet_unex) > 0:
            raise ValueError(f"UNet has unexpected keys: {unet_unex}")
        tenc_missing, _ = text_encoder.load_state_dict(tenc_state_dict, strict=False)
        if len(tenc_missing) > 0:
            raise ValueError(f"TextEncoder has missing keys: {tenc_missing}")
        vae_missing, _ = vae.load_state_dict(vae_state_dict, strict=False)
        if len(vae_missing) > 0:
            raise ValueError(f"VAE has missing keys: {vae_missing}")
    else:
        logger.info("Using base model weights (no checkpoint/LoRA)")

    # enable xformers if available
    if use_xformers:
        logger.info("Enabling xformers memory-efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    # lora
    for l in model_config.lora_map:
        lora_path = data_dir.joinpath(l)
        if lora_path.is_file():
            logger.info(f"Loading lora {lora_path}")
            logger.info(f"alpha = {model_config.lora_map[l]}")
            load_safetensors_lora(text_encoder, unet, lora_path, alpha=model_config.lora_map[l])

    logger.info("Creating AnimationPipeline...")
    pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )

    # Load TI embeddings
    load_text_embeddings(pipeline)

    return pipeline

def create_us_pipeline(
    model_config: ModelConfig = ...,
    infer_config: InferenceConfig = ...,
    use_xformers: bool = True,
) -> DiffusionPipeline:

    # set up scheduler
    sched_kwargs = infer_config.noise_scheduler_kwargs
    scheduler = get_scheduler(model_config.scheduler, sched_kwargs)
    logger.info(f'Using scheduler "{model_config.scheduler}" ({scheduler.__class__.__name__})')

    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile')

    # Load the checkpoint weights into the pipeline
    pipeline:DiffusionPipeline

    if model_config.path is not None:
        model_path = data_dir.joinpath(model_config.path)
        logger.info(f"Loading weights from {model_path}")
        if model_path.is_file():

            def is_empty_dir(path):
                import os
                return len(os.listdir(path)) == 0

            save_path = data_dir.joinpath("models/huggingface/" + model_path.stem + "_" + str(model_path.stat().st_size))
            save_path.mkdir(exist_ok=True)
            if save_path.is_dir() and is_empty_dir(save_path):
                # StableDiffusionControlNetImg2ImgPipeline.from_single_file does not exist in version 18.2
                logger.debug("Loading from single checkpoint file")
                tmp_pipeline = StableDiffusionPipeline.from_single_file(
                    pretrained_model_link_or_path=str(model_path.absolute())
                )
                tmp_pipeline.save_pretrained(save_path, safe_serialization=True)
                del tmp_pipeline

            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                save_path,
                controlnet=controlnet,
                local_files_only=False,
                load_safety_checker=False,
                safety_checker=None,
            )

        elif model_path.is_dir():
            logger.debug("Loading from Diffusers model directory")
            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                model_path,
                controlnet=controlnet,
                local_files_only=True,
                load_safety_checker=False,
                safety_checker=None,
            )
        else:
            raise FileNotFoundError(f"model_path {model_path} is not a file or directory")
    else:
        ValueError("model_config.path is invalid")

    pipeline.scheduler = scheduler

    # enable xformers if available
    if use_xformers:
        logger.info("Enabling xformers memory-efficient attention")
        pipeline.enable_xformers_memory_efficient_attention()

    # lora
    for l in model_config.lora_map:
        lora_path = data_dir.joinpath(l)
        if lora_path.is_file():
            logger.info(f"Loading lora {lora_path}")
            logger.info(f"alpha = {model_config.lora_map[l]}")
            load_safetensors_lora(pipeline.text_encoder, pipeline.unet, lora_path, alpha=model_config.lora_map[l],is_animatediff=False)

    # Load TI embeddings
    load_text_embeddings(pipeline)

    return pipeline


def seed_everything(seed):
    import random

    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def run_inference(
    pipeline: AnimationPipeline,
    prompt: str = ...,
    n_prompt: str = ...,
    seed: int = -1,
    steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    duration: int = 16,
    idx: int = 0,
    out_dir: PathLike = ...,
    context_frames: int = -1,
    context_stride: int = 3,
    context_overlap: int = 4,
    context_schedule: str = "uniform",
    clip_skip: int = 1,
    return_dict: bool = False,
    prompt_map: Dict[int, str] = None,
):
    out_dir = Path(out_dir)  # ensure out_dir is a Path

    if seed == -1:
        seed = torch.seed()

    seed_everything(seed)

    pipeline_output = pipeline(
        prompt=prompt,
        negative_prompt=n_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        video_length=duration,
        return_dict=return_dict,
        context_frames=context_frames,
        context_stride=context_stride + 1,
        context_overlap=context_overlap,
        context_schedule=context_schedule,
        clip_skip=clip_skip,
        prompt_map=prompt_map,
    )
    logger.info("Generation complete, saving...")

    # Trim and clean up the prompt for filename use
    prompt_tags = [re_clean_prompt.sub("", tag).strip().replace(" ", "-") for tag in prompt_map[list(prompt_map.keys())[0]].split(",")]
    prompt_str = "_".join((prompt_tags[:6]))

    # generate the output filename and save the video
    out_file = out_dir.joinpath(f"{idx:02d}_{seed}_{prompt_str}.gif")
    if return_dict is True:
        save_video(pipeline_output["videos"], out_file)
    else:
        save_video(pipeline_output, out_file)

    logger.info(f"Saved sample to {out_file}")
    return pipeline_output


def run_upscale(
    org_imgs: List[str],
    pipeline: DiffusionPipeline,
    prompt_map: Dict[int, str] = None,
    n_prompt: str = ...,
    seed: int = -1,
    steps: int = 25,
    strength: float = 0.5,
    guidance_scale: float = 7.5,
    clip_skip: int = 1,
    us_width: int = 512,
    us_height: int = 512,
    idx: int = 0,
    out_dir: PathLike = ...,
    upscale_config:Dict[str, Any]=None,
):
    from animatediff.utils.lpw_stable_diffusion import lpw_encode_prompt

    if us_width < 0 and us_height < 0:
        logger.info(f"invalid width,height: {us_width},{us_height}")
        return None

    pipeline.set_progress_bar_config(disable=True)

    images = get_resized_images(org_imgs, us_width, us_height)

    steps = steps if "steps" not in upscale_config else upscale_config["steps"]
    scheduler = scheduler if "scheduler" not in upscale_config else upscale_config["scheduler"]
    guidance_scale = guidance_scale if "guidance_scale" not in upscale_config else upscale_config["guidance_scale"]
    clip_skip = clip_skip if "clip_skip" not in upscale_config else upscale_config["clip_skip"]
    strength = strength if "strength" not in upscale_config else upscale_config["strength"]

    generator = torch.manual_seed(seed)

    seed_everything(seed)

    prompt_embeds_map = {}
    prompt_map = dict(sorted(prompt_map.items()))
    negative = None

    do_classifier_free_guidance=guidance_scale > 1.0

    prompt_list = [prompt_map[key_frame] for key_frame in prompt_map.keys()]

    prompt_embeds,neg_embeds = lpw_encode_prompt(
        pipe=pipeline,
        prompt=prompt_list,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=n_prompt,
    )

    if do_classifier_free_guidance:
        negative = neg_embeds.chunk(neg_embeds.shape[0], 0)
        positive = prompt_embeds.chunk(prompt_embeds.shape[0], 0)
    else:
        negative = [None]
        positive = prompt_embeds.chunk(prompt_embeds.shape[0], 0)

    for i, key_frame in enumerate(prompt_map):
        prompt_embeds_map[key_frame] = positive[i]

    key_first =list(prompt_map.keys())[0]
    key_last =list(prompt_map.keys())[-1]

    def get_current_prompt_embeds(
            center_frame: int = 0,
            video_length : int = 0
            ):

        key_prev = key_last
        key_next = key_first

        for p in prompt_map.keys():
            if p > center_frame:
                key_next = p
                break
            key_prev = p

        dist_prev = center_frame - key_prev
        if dist_prev < 0:
            dist_prev += video_length
        dist_next = key_next - center_frame
        if dist_next < 0:
            dist_next += video_length

        if key_prev == key_next or dist_prev + dist_next == 0:
            return prompt_embeds_map[key_prev]

        rate = dist_prev / (dist_prev + dist_next)

        return prompt_embeds_map[key_prev] * (1-rate) + prompt_embeds_map[key_next] * (rate)

    out_images=[]

    for i, condition_image in enumerate(tqdm(images, desc=f"Upscaling...")):

        cur_positive = get_current_prompt_embeds(i, len(images))

#        logger.info(f"w {condition_image.size[0]}")
#        logger.info(f"h {condition_image.size[1]}")

        out_image = pipeline(
            prompt_embeds=cur_positive,
            negative_prompt_embeds=negative[0],
            image=condition_image,
            control_image=condition_image,
            width=condition_image.size[0],
            height=condition_image.size[1],
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        out_images.append(out_image)

    # Trim and clean up the prompt for filename use
    prompt_tags = [re_clean_prompt.sub("", tag).strip().replace(" ", "-") for tag in prompt_map[list(prompt_map.keys())[0]].split(",")]
    prompt_str = "_".join((prompt_tags[:6]))

    # generate the output filename and save the video
    out_file = out_dir.joinpath(f"{idx:02d}_{seed}_{prompt_str}.gif")

    out_images[0].save(
        fp=out_file, format="GIF", append_images=out_images[1:], save_all=True, duration=(1 / 8 * 1000), loop=0
    )

    logger.info(f"Saved sample to {out_file}")

    return out_images

