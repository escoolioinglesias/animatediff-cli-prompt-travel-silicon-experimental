import glob
import logging
import os
import re
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from controlnet_aux import HEDdetector, LineartAnimeDetector, OpenposeDetector
from diffusers import (AutoencoderKL, ControlNetModel, DiffusionPipeline,
                       StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionPipeline)
from tqdm.rich import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from animatediff import get_dir
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.pipelines.pipeline_controlnet_img2img_reference import \
    StableDiffusionControlNetImg2ImgReferencePipeline
from animatediff.schedulers import get_scheduler
from animatediff.settings import InferenceConfig, ModelConfig
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatediff.utils.model import (ensure_motion_modules,
                                     get_checkpoint_weights)
from animatediff.utils.util import (get_resized_image, get_resized_images,
                                    save_video)

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
default_base_path = data_dir.joinpath("models/huggingface/stable-diffusion-v1-5")

re_clean_prompt = re.compile(r"[^\w\-, ]")

lineart_anime_processor = None
openpose_processor=None
softedge_processor=None

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


def create_controlnet_model(type_str):
    if type_str == "controlnet_tile":
        return ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile')
    elif type_str == "controlnet_lineart_anime":
        return ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15s2_lineart_anime')
    elif type_str == "controlnet_ip2p":
        return ControlNetModel.from_pretrained('lllyasviel/control_v11e_sd15_ip2p')
    elif type_str == "controlnet_openpose":
        return ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose')
    elif type_str == "controlnet_softedge":
        return ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge')
    else:
        raise ValueError(f"unknown controlnet type {type_str}")



def get_preprocessor(type_str):
    global lineart_anime_processor,openpose_processor,softedge_processor
    if type_str == "controlnet_lineart_anime":
        if not lineart_anime_processor:
            lineart_anime_processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        return lineart_anime_processor
    elif type_str == "controlnet_openpose":
        if not openpose_processor:
            openpose_processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        return openpose_processor
    elif type_str == "controlnet_softedge":
        if not softedge_processor:
            softedge_processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
        return softedge_processor
    else:
        raise ValueError(f"unknown controlnet type {type_str}")


def get_preprocessed_img(type_str, img):
    if type_str in ( "controlnet_tile", "controlnet_ip2p"):
        return img
    elif type_str in ( "controlnet_lineart_anime" , "controlnet_openpose" ,"controlnet_softedge"):
        return get_preprocessor(type_str)(img)
    else:
        raise ValueError(f"unknown controlnet type {type_str}")



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

    # controlnet
    controlnet_map={}
    if model_config.controlnet_map:
        c_image_dir = data_dir.joinpath( model_config.controlnet_map["input_image_dir"] )

        for c in model_config.controlnet_map:
            item = model_config.controlnet_map[c]
            if type(item) is dict:
                if item["enable"] == True:
                    img_dir = c_image_dir.joinpath( c )
                    cond_imgs = sorted(glob.glob( os.path.join(img_dir, "[0-9]*.png"), recursive=False))
                    if len(cond_imgs) > 0:
                        controlnet_map[c] = create_controlnet_model( c )

    if not controlnet_map:
        controlnet_map = None

    logger.info("Creating AnimationPipeline...")
    pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        controlnet_map=controlnet_map,
    )

    # Load TI embeddings
    load_text_embeddings(pipeline)

    return pipeline

def create_us_pipeline(
    model_config: ModelConfig = ...,
    infer_config: InferenceConfig = ...,
    use_xformers: bool = True,
    use_controlnet_ref: bool = False,
    use_controlnet_tile: bool = False,
    use_controlnet_line_anime: bool = False,
    use_controlnet_ip2p: bool = False,
) -> DiffusionPipeline:

    # set up scheduler
    sched_kwargs = infer_config.noise_scheduler_kwargs
    scheduler = get_scheduler(model_config.scheduler, sched_kwargs)
    logger.info(f'Using scheduler "{model_config.scheduler}" ({scheduler.__class__.__name__})')

    controlnet = []
    if use_controlnet_tile:
        controlnet.append( ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile') )
    if use_controlnet_line_anime:
        controlnet.append( ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15s2_lineart_anime') )
    if use_controlnet_ip2p:
        controlnet.append( ControlNetModel.from_pretrained('lllyasviel/control_v11e_sd15_ip2p') )

    if len(controlnet) == 1:
        controlnet = controlnet[0]

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

            if use_controlnet_ref:
                pipeline = StableDiffusionControlNetImg2ImgReferencePipeline.from_pretrained(
                    save_path,
                    controlnet=controlnet,
                    local_files_only=False,
                    load_safety_checker=False,
                    safety_checker=None,
                )
            else:
                pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    save_path,
                    controlnet=controlnet,
                    local_files_only=False,
                    load_safety_checker=False,
                    safety_checker=None,
                )

        elif model_path.is_dir():
            logger.debug("Loading from Diffusers model directory")
            if use_controlnet_ref:
                pipeline = StableDiffusionControlNetImg2ImgReferencePipeline.from_pretrained(
                    model_path,
                    controlnet=controlnet,
                    local_files_only=True,
                    load_safety_checker=False,
                    safety_checker=None,
                )
            else:
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
        raise ValueError("model_config.path is invalid")

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
    controlnet_map: Dict[str, Any] = None,
):
    out_dir = Path(out_dir)  # ensure out_dir is a Path

    controlnet_type_map={}
    controlnet_image_map={}
    # { 0 : { "type_str" : IMAGE, "type_str2" : IMAGE }  }

    if controlnet_map:
        c_image_dir = data_dir.joinpath( controlnet_map["input_image_dir"] )

        for c in controlnet_map:
            item = controlnet_map[c]
            if type(item) is dict:
                if item["enable"] == True:
                    img_dir = c_image_dir.joinpath( c )
                    cond_imgs = sorted(glob.glob( os.path.join(img_dir, "[0-9]*.png"), recursive=False))
                    if len(cond_imgs) > 0:
                        controlnet_type_map[c] = {
                            "controlnet_conditioning_scale" : item["controlnet_conditioning_scale"],
                            "control_guidance_start" : item["control_guidance_start"],
                            "control_guidance_end" : item["control_guidance_end"],
                            "control_scale_list" : item["control_scale_list"],
                        }
                    for img_path in cond_imgs:
                        frame_no = int(Path(img_path).stem)
                        if frame_no not in controlnet_image_map:
                            controlnet_image_map[frame_no] = {}
                        controlnet_image_map[frame_no][c] = get_preprocessed_img( c, get_resized_image(img_path, width, height) )


    if not controlnet_type_map:
        controlnet_type_map=None
    if not controlnet_image_map:
        controlnet_image_map=None

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
        controlnet_type_map=controlnet_type_map,
        controlnet_image_map=controlnet_image_map,
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
    use_controlnet_ref: bool = False,
    use_controlnet_tile: bool = False,
    use_controlnet_line_anime: bool = False,
    use_controlnet_ip2p: bool = False,
):
    from animatediff.utils.lpw_stable_diffusion import lpw_encode_prompt

    pipeline.set_progress_bar_config(disable=True)

    images = get_resized_images(org_imgs, us_width, us_height)

    steps = steps if "steps" not in upscale_config else upscale_config["steps"]
    scheduler = scheduler if "scheduler" not in upscale_config else upscale_config["scheduler"]
    guidance_scale = guidance_scale if "guidance_scale" not in upscale_config else upscale_config["guidance_scale"]
    clip_skip = clip_skip if "clip_skip" not in upscale_config else upscale_config["clip_skip"]
    strength = strength if "strength" not in upscale_config else upscale_config["strength"]

    controlnet_conditioning_scale = []
    guess_mode = []
    control_guidance_start = []
    control_guidance_end = []

    # for controlnet tile
    if use_controlnet_tile:
        controlnet_conditioning_scale.append(upscale_config["controlnet_tile"]["controlnet_conditioning_scale"])
        guess_mode.append(upscale_config["controlnet_tile"]["guess_mode"])
        control_guidance_start.append(upscale_config["controlnet_tile"]["control_guidance_start"])
        control_guidance_end.append(upscale_config["controlnet_tile"]["control_guidance_end"])

    # for controlnet line_anime
    if use_controlnet_line_anime:
        controlnet_conditioning_scale.append(upscale_config["controlnet_line_anime"]["controlnet_conditioning_scale"])
        guess_mode.append(upscale_config["controlnet_line_anime"]["guess_mode"])
        control_guidance_start.append(upscale_config["controlnet_line_anime"]["control_guidance_start"])
        control_guidance_end.append(upscale_config["controlnet_line_anime"]["control_guidance_end"])

    # for controlnet ip2p
    if use_controlnet_ip2p:
        controlnet_conditioning_scale.append(upscale_config["controlnet_ip2p"]["controlnet_conditioning_scale"])
        guess_mode.append(upscale_config["controlnet_ip2p"]["guess_mode"])
        control_guidance_start.append(upscale_config["controlnet_ip2p"]["control_guidance_start"])
        control_guidance_end.append(upscale_config["controlnet_ip2p"]["control_guidance_end"])

    # for controlnet ref
    ref_image = None
    if use_controlnet_ref:
        if not upscale_config["controlnet_ref"]["use_frame_as_ref_image"] and not upscale_config["controlnet_ref"]["use_1st_frame_as_ref_image"]:
            ref_image = get_resized_images([ data_dir.joinpath( upscale_config["controlnet_ref"]["ref_image"] ) ], us_width, us_height)[0]


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


    line_anime_processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")


    out_images=[]

    logger.info(f"{use_controlnet_tile=}")
    logger.info(f"{use_controlnet_line_anime=}")
    logger.info(f"{use_controlnet_ip2p=}")

    logger.info(f"{controlnet_conditioning_scale=}")
    logger.info(f"{guess_mode=}")
    logger.info(f"{control_guidance_start=}")
    logger.info(f"{control_guidance_end=}")


    for i, org_image in enumerate(tqdm(images, desc=f"Upscaling...")):

        cur_positive = get_current_prompt_embeds(i, len(images))

#        logger.info(f"w {condition_image.size[0]}")
#        logger.info(f"h {condition_image.size[1]}")
        condition_image = []

        if use_controlnet_tile:
            condition_image.append( org_image )
        if use_controlnet_line_anime:
            condition_image.append( line_anime_processor(org_image) )
        if use_controlnet_ip2p:
            condition_image.append( org_image )

        if not use_controlnet_ref:
            out_image = pipeline(
                prompt_embeds=cur_positive,
                negative_prompt_embeds=negative[0],
                image=org_image,
                control_image=condition_image,
                width=org_image.size[0],
                height=org_image.size[1],
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,

                controlnet_conditioning_scale= controlnet_conditioning_scale if len(controlnet_conditioning_scale) > 1 else controlnet_conditioning_scale[0],
                guess_mode= guess_mode[0],
                control_guidance_start= control_guidance_start if len(control_guidance_start) > 1 else control_guidance_start[0],
                control_guidance_end= control_guidance_end if len(control_guidance_end) > 1 else control_guidance_end[0],

            ).images[0]
        else:

            if upscale_config["controlnet_ref"]["use_1st_frame_as_ref_image"]:
                if i == 0:
                    ref_image = org_image
            elif upscale_config["controlnet_ref"]["use_frame_as_ref_image"]:
                ref_image = org_image

            out_image = pipeline(
                prompt_embeds=cur_positive,
                negative_prompt_embeds=negative[0],
                image=org_image,
                control_image=condition_image,
                width=org_image.size[0],
                height=org_image.size[1],
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,

                controlnet_conditioning_scale= controlnet_conditioning_scale if len(controlnet_conditioning_scale) > 1 else controlnet_conditioning_scale[0],
                guess_mode= guess_mode[0],
                # control_guidance_start= control_guidance_start,
                # control_guidance_end= control_guidance_end,

                ### for controlnet ref
                ref_image=ref_image,
                attention_auto_machine_weight = upscale_config["controlnet_ref"]["attention_auto_machine_weight"],
                gn_auto_machine_weight = upscale_config["controlnet_ref"]["gn_auto_machine_weight"],
                style_fidelity = upscale_config["controlnet_ref"]["style_fidelity"],
                reference_attn= upscale_config["controlnet_ref"]["reference_attn"],
                reference_adain= upscale_config["controlnet_ref"]["reference_adain"],

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

