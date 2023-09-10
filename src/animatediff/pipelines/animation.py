# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import (BaseOutput, deprecate, is_accelerate_available,
                             is_accelerate_version, is_compiled_module,
                             randn_tensor)
from einops import rearrange
from packaging import version
from tqdm.rich import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from animatediff.ip_adapter import IPAdapter, IPAdapterPlus
from animatediff.models.attention import BasicTransformerBlock
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import (UNet3DConditionModel,
                                     UNetMidBlock3DCrossAttn)
from animatediff.models.unet_blocks import (CrossAttnDownBlock3D,
                                            CrossAttnUpBlock3D, DownBlock3D,
                                            UpBlock3D)
from animatediff.pipelines.context import (get_context_scheduler,
                                           get_total_steps)
from animatediff.utils.model import nop_train
from animatediff.utils.pipeline import get_memory_format
from animatediff.utils.util import (end_profile, show_gpu, start_profile,
                                    stopwatch_record, stopwatch_start,
                                    stopwatch_stop)

logger = logging.getLogger(__name__)



C_REF_MODE = "write"


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
            result += torch_dfs(child)
    return result


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    _optional_components = ["feature_extractor"]

    vae: AutoencoderKL
    text_encoder: CLIPSkipTextModel
    tokenizer: CLIPTokenizer
    unet: UNet3DConditionModel
    feature_extractor: CLIPImageProcessor
    scheduler: Union[
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    ]
    controlnet_map: Dict[ str , ControlNetModel ]
    ip_adapter: IPAdapter = None

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPSkipTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        feature_extractor: CLIPImageProcessor,
        controlnet_map: Dict[ str , ControlNetModel ]=None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.controlnet_map = controlnet_map


    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt=None,
        max_embeddings_multiples=3,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: int = 1,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        from ..utils.lpw_stable_diffusion import get_weighted_text_embeddings

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
        if prompt_embeds is None or negative_prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if do_classifier_free_guidance and negative_prompt_embeds is None:
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, self.tokenizer)

            prompt_embeds1, negative_prompt_embeds1 = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=clip_skip
            )
            if prompt_embeds is None:
                prompt_embeds = prompt_embeds1
            if negative_prompt_embeds is None:
                negative_prompt_embeds = negative_prompt_embeds1

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def __encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: int = 1,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                clip_skip=clip_skip,
            )
            prompt_embeds = prompt_embeds[0]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids.to(device),
                attention_mask=attention_mask,
                clip_skip=clip_skip,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents: torch.Tensor):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(
                self.vae.decode(latents[frame_idx : frame_idx + 1].to(self.vae.device, self.vae.dtype)).sample
            )
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_ref_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.unet.device, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents.to(device, dtype)


    # from diffusers/examples/community/stable_diffusion_controlnet_reference.py
    def prepare_ref_latents(self, refimage, batch_size, dtype, device, generator, do_classifier_free_guidance):
        refimage = refimage.to(device=device, dtype=self.vae.dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(generator=generator)
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        ref_image_latents = torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents


    # from diffusers/examples/community/stable_diffusion_controlnet_reference.py
    def prepare_controlnet_ref_only_without_motion(
        self,
        ref_image_latents,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
    ):
        global C_REF_MODE
        # 9. Modify self attention and group norm
        C_REF_MODE = "write"
        uc_mask = (
            torch.Tensor([1] * batch_size * num_images_per_prompt + [0] * batch_size * num_images_per_prompt)
            .type_as(ref_image_latents)
            .bool()
        )


        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            video_length=None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.unet_use_cross_frame_attention:
                cross_attention_kwargs["video_length"] = video_length

            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if C_REF_MODE == "write":
                    self.bank.append(norm_hidden_states.detach().clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if C_REF_MODE == "read":
                    if attention_auto_machine_weight > self.attn_weight:
                        attn_output_uc = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                            # attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )

                        if style_fidelity > 0:
                            attn_output_c = attn_output_uc.clone()

                            if do_classifier_free_guidance:
                                attn_output_c[uc_mask] = self.attn1(
                                    norm_hidden_states[uc_mask],
                                    encoder_hidden_states=norm_hidden_states[uc_mask],
                                    **cross_attention_kwargs,
                                )

                            attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc

                        else:
                            attn_output = attn_output_uc

                        self.bank.clear()
                    else:
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )

            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

            # 4. Temporal-Attention
            if self.unet_use_temporal_attention:
                d = hidden_states.shape[1]
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                norm_hidden_states = (
                    self.norm_temp(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm_temp(hidden_states)
                )
                hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states

        def hacked_mid_forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ) -> torch.FloatTensor:

            eps = 1e-6

            hidden_states = self.resnets[0](hidden_states, temb)
            for attn, resnet, motion_module in zip(self.attentions, self.resnets[1:], self.motion_modules):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                x = hidden_states

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(x, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(x, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                        var_acc = sum(self.var_bank) / float(len(self.var_bank))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        x_uc = (((x - mean) / std) * std_acc) + mean_acc
                        x_c = x_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = x.shape[2]
                            x_c = rearrange(x_c, "b c f h w -> (b f) c h w")
                            x = rearrange(x, "b c f h w -> (b f) c h w")

                            x_c[uc_mask] = x[uc_mask]

                            x_c = rearrange(x_c, "(b f) c h w -> b c f h w", f=f)
                            x = rearrange(x, "(b f) c h w -> b c f h w", f=f)

                        x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
                    self.mean_bank = []
                    self.var_bank = []

                hidden_states = x

                if motion_module is not None:
                    hidden_states = motion_module(
                        hidden_states,
                        temb,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                hidden_states = resnet(hidden_states, temb)

            return hidden_states


        def hack_CrossAttnDownBlock3D_forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6

            # TODO(Patrick, William) - attention mask is not used
            output_states = ()

            for i, (resnet, attn, motion_module) in enumerate(zip(self.resnets, self.attentions, self.motion_modules)):
                hidden_states = resnet(hidden_states, temb)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                # add motion module

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                hidden_states = (
                    motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                    if motion_module is not None
                    else hidden_states
                )

                output_states = output_states + (hidden_states,)

            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_DownBlock3D_forward(self, hidden_states, temb=None, encoder_hidden_states=None):
            eps = 1e-6

            output_states = ()

            for i, (resnet, motion_module) in enumerate(zip(self.resnets, self.motion_modules)):
                hidden_states = resnet(hidden_states, temb)

                # add motion module

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                if motion_module:
                    hidden_states = motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )

                output_states = output_states + (hidden_states,)

            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_CrossAttnUpBlock3D_forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6
            # TODO(Patrick, William) - attention mask is not used
            for i, (resnet, attn, motion_module) in enumerate(zip(self.resnets, self.attentions, self.motion_modules)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # add motion module

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                if motion_module:
                    hidden_states = motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )


            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        def hacked_UpBlock3D_forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, encoder_hidden_states=None):
            eps = 1e-6
            for i, (resnet,motion_module) in enumerate(zip(self.resnets, self.motion_modules)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                if motion_module:
                    hidden_states = motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )



            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        if reference_attn:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

            attn_modules = None
            torch.cuda.empty_cache()

        if reference_adain:
            gn_modules = [self.unet.mid_block]
            self.unet.mid_block.gn_weight = 0

            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                if getattr(module, "original_forward", None) is None:
                    module.original_forward = module.forward
                if i == 0:
                    # mid_block
                    module.forward = hacked_mid_forward.__get__(module, UNetMidBlock3DCrossAttn)
                elif isinstance(module, CrossAttnDownBlock3D):
                    module.forward = hack_CrossAttnDownBlock3D_forward.__get__(module, CrossAttnDownBlock3D)
                elif isinstance(module, DownBlock3D):
                    module.forward = hacked_DownBlock3D_forward.__get__(module, DownBlock3D)
                elif isinstance(module, CrossAttnUpBlock3D):
                    module.forward = hacked_CrossAttnUpBlock3D_forward.__get__(module, CrossAttnUpBlock3D)
                elif isinstance(module, UpBlock3D):
                    module.forward = hacked_UpBlock3D_forward.__get__(module, UpBlock3D)
                module.mean_bank = []
                module.var_bank = []
                module.gn_weight *= 2

            gn_modules = None
            torch.cuda.empty_cache()



    # from diffusers/examples/community/stable_diffusion_controlnet_reference.py
    def prepare_controlnet_ref_only(
        self,
        ref_image_latents,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        _scale_pattern,
    ):
        global C_REF_MODE
        # 9. Modify self attention and group norm
        C_REF_MODE = "write"
        uc_mask = (
            torch.Tensor([1] * batch_size * num_images_per_prompt + [0] * batch_size * num_images_per_prompt)
            .type_as(ref_image_latents)
            .bool()
        )

        _scale_pattern = _scale_pattern * (batch_size // len(_scale_pattern) + 1)
        _scale_pattern = _scale_pattern[:batch_size]
        _rev_pattern = [1-i for i in _scale_pattern]

        scale_pattern_double = torch.tensor(_scale_pattern*2).to(self.device, dtype=self.unet.dtype)
        rev_pattern_double = torch.tensor(_rev_pattern*2).to(self.device, dtype=self.unet.dtype)
        scale_pattern = torch.tensor(_scale_pattern).to(self.device, dtype=self.unet.dtype)
        rev_pattern = torch.tensor(_rev_pattern).to(self.device, dtype=self.unet.dtype)



        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            video_length=None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.unet_use_cross_frame_attention:
                cross_attention_kwargs["video_length"] = video_length

            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if C_REF_MODE == "write":
                    self.bank.append(norm_hidden_states.detach().clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if C_REF_MODE == "read":
                    if attention_auto_machine_weight > self.attn_weight:
                        attn_output_uc = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                            # attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )

                        if style_fidelity > 0:
                            attn_output_c = attn_output_uc.clone()

                            if do_classifier_free_guidance:
                                attn_output_c[uc_mask] = self.attn1(
                                    norm_hidden_states[uc_mask],
                                    encoder_hidden_states=norm_hidden_states[uc_mask],
                                    **cross_attention_kwargs,
                                )

                            attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc

                        else:
                            attn_output = attn_output_uc

                        attn_org = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )

                        attn_output = scale_pattern_double[:,None,None] * attn_output + rev_pattern_double[:,None,None] * attn_org

                        self.bank.clear()
                    else:
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )

            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

            # 4. Temporal-Attention
            if self.unet_use_temporal_attention:
                d = hidden_states.shape[1]
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                norm_hidden_states = (
                    self.norm_temp(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm_temp(hidden_states)
                )
                hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states

        def hacked_mid_forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ) -> torch.FloatTensor:

            eps = 1e-6

            hidden_states = self.resnets[0](hidden_states, temb)
            for attn, resnet, motion_module in zip(self.attentions, self.resnets[1:], self.motion_modules):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if motion_module is not None:
                    hidden_states = motion_module(
                        hidden_states,
                        temb,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                hidden_states = resnet(hidden_states, temb)

                x = hidden_states

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(x, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(x, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                        var_acc = sum(self.var_bank) / float(len(self.var_bank))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        x_uc = (((x - mean) / std) * std_acc) + mean_acc
                        x_c = x_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = x.shape[2]
                            x_c = rearrange(x_c, "b c f h w -> (b f) c h w")
                            x = rearrange(x, "b c f h w -> (b f) c h w")

                            x_c[uc_mask] = x[uc_mask]

                            x_c = rearrange(x_c, "(b f) c h w -> b c f h w", f=f)
                            x = rearrange(x, "(b f) c h w -> b c f h w", f=f)

                        mod_x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc

                    x = scale_pattern[None,None,:,None,None] * mod_x + rev_pattern[None,None,:,None,None] * x

                    self.mean_bank = []
                    self.var_bank = []

                hidden_states = x

            return hidden_states

        def hack_CrossAttnDownBlock3D_forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6

            # TODO(Patrick, William) - attention mask is not used
            output_states = ()

            for i, (resnet, attn, motion_module) in enumerate(zip(self.resnets, self.attentions, self.motion_modules)):
                hidden_states = resnet(hidden_states, temb)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                # add motion module
                hidden_states = (
                    motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                    if motion_module is not None
                    else hidden_states
                )

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        mod_hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                        hidden_states = scale_pattern[None,None,:,None,None] * mod_hidden_states + rev_pattern[None,None,:,None,None] * hidden_states

                output_states = output_states + (hidden_states,)

            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_DownBlock3D_forward(self, hidden_states, temb=None, encoder_hidden_states=None):
            eps = 1e-6

            output_states = ()

            for i, (resnet, motion_module) in enumerate(zip(self.resnets, self.motion_modules)):
                hidden_states = resnet(hidden_states, temb)

                # add motion module
                if motion_module:
                    hidden_states = motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        mod_hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                        hidden_states = scale_pattern[None,None,:,None,None] * mod_hidden_states + rev_pattern[None,None,:,None,None] * hidden_states

                output_states = output_states + (hidden_states,)

            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_CrossAttnUpBlock3D_forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6
            # TODO(Patrick, William) - attention mask is not used
            for i, (resnet, attn, motion_module) in enumerate(zip(self.resnets, self.attentions, self.motion_modules)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # add motion module
                if motion_module:
                    hidden_states = motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:

                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        mod_hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                        hidden_states = scale_pattern[None,None,:,None,None] * mod_hidden_states + rev_pattern[None,None,:,None,None] * hidden_states


            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        def hacked_UpBlock3D_forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, encoder_hidden_states=None):
            eps = 1e-6
            for i, (resnet,motion_module) in enumerate(zip(self.resnets, self.motion_modules)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if motion_module:
                    hidden_states = motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )

                if C_REF_MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if C_REF_MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(3, 4), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            f = hidden_states.shape[2]
                            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                            hidden_states_c = rearrange(hidden_states_c, "b c f h w -> (b f) c h w")

                            hidden_states_c[uc_mask] = hidden_states[uc_mask]

                            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
                            hidden_states_c = rearrange(hidden_states_c, "(b f) c h w -> b c f h w", f=f)

                        mod_hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                        hidden_states = scale_pattern[None,None,:,None,None] * mod_hidden_states + rev_pattern[None,None,:,None,None] * hidden_states

            if C_REF_MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        if reference_attn:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

            attn_modules = None
            torch.cuda.empty_cache()

        if reference_adain:
            gn_modules = [self.unet.mid_block]
            self.unet.mid_block.gn_weight = 0

            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                if getattr(module, "original_forward", None) is None:
                    module.original_forward = module.forward
                if i == 0:
                    # mid_block
                    module.forward = hacked_mid_forward.__get__(module, UNetMidBlock3DCrossAttn)
                elif isinstance(module, CrossAttnDownBlock3D):
                    module.forward = hack_CrossAttnDownBlock3D_forward.__get__(module, CrossAttnDownBlock3D)
                elif isinstance(module, DownBlock3D):
                    module.forward = hacked_DownBlock3D_forward.__get__(module, DownBlock3D)
                elif isinstance(module, CrossAttnUpBlock3D):
                    module.forward = hacked_CrossAttnUpBlock3D_forward.__get__(module, CrossAttnUpBlock3D)
                elif isinstance(module, UpBlock3D):
                    module.forward = hacked_UpBlock3D_forward.__get__(module, UpBlock3D)
                module.mean_bank = []
                module.var_bank = []
                module.gn_weight *= 2

            gn_modules = None
            torch.cuda.empty_cache()


    def unload_controlnet_ref_only(
        self,
        reference_attn,
        reference_adain,
    ):
        if reference_attn:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module.forward = module._original_inner_forward
                module.bank = []

            attn_modules = None
            torch.cuda.empty_cache()

        if reference_adain:
            gn_modules = [self.unet.mid_block]
            self.unet.mid_block.gn_weight = 0

            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                module.forward = module.original_forward
                module.mean_bank = []
                module.var_bank = []
                module.gn_weight *= 2

            gn_modules = None
            torch.cuda.empty_cache()


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        video_length: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        context_frames: int = -1,
        context_stride: int = 3,
        context_overlap: int = 4,
        context_schedule: str = "uniform",
        clip_skip: int = 1,
        prompt_map: Dict[int, str] = None,
        controlnet_type_map: Dict[str, Dict[str,float]] = None,
        controlnet_image_map: Dict[int, Dict[str,Any]] = None,
        controlnet_ref_map: Dict[str, Any] = None,
        controlnet_max_samples_on_vram: int = 999,
        controlnet_max_models_on_vram: int=99,
        controlnet_is_loop: bool=True,
        ip_adapter_map: Dict[str, Any] = None,
        **kwargs,
    ):
        global C_REF_MODE

        controlnet_image_map_org = controlnet_image_map

        controlnet_max_models_on_vram = max(controlnet_max_models_on_vram,1)

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 16 frames is max reliable number for one-shot mode, so we use sequential mode for longer videos
        sequential_mode = video_length is not None and video_length > 48

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # Define call parameters
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        latents_device = torch.device("cpu") if sequential_mode else device


        if ip_adapter_map:
            if self.ip_adapter is None:
                img_enc_path = "data/models/ip_adapter/models/image_encoder/"
                if ip_adapter_map["is_plus_face"]:
                    self.ip_adapter = IPAdapterPlus(self, img_enc_path, "data/models/ip_adapter/models/ip-adapter-plus-face_sd15.bin", device, 16)
                elif ip_adapter_map["is_plus"]:
                    self.ip_adapter = IPAdapterPlus(self, img_enc_path, "data/models/ip_adapter/models/ip-adapter-plus_sd15.bin", device, 16)
                else:
                    self.ip_adapter = IPAdapter(self, img_enc_path, "data/models/ip_adapter/models/ip-adapter_sd15.bin", device, 4)
                self.ip_adapter.set_scale( ip_adapter_map["scale"] )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        ### text
        prompt_embeds_map = {}
        prompt_map = dict(sorted(prompt_map.items()))

        prompt_list = [prompt_map[key_frame] for key_frame in prompt_map.keys()]
        prompt_embeds = self._encode_prompt(
            prompt_list,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            clip_skip=clip_skip,
        )

        if do_classifier_free_guidance:
            negative, positive = prompt_embeds.chunk(2, 0)
            negative = negative.chunk(negative.shape[0], 0)
            positive = positive.chunk(positive.shape[0], 0)
        else:
            positive = prompt_embeds
            positive = positive.chunk(positive.shape[0], 0)

        if self.ip_adapter:
            self.ip_adapter.set_text_length(positive[0].shape[1])

        for i, key_frame in enumerate(prompt_map):
            if do_classifier_free_guidance:
                prompt_embeds_map[key_frame] = torch.cat([negative[i] , positive[i]])
            else:
                prompt_embeds_map[key_frame] = positive[i]

        key_first =list(prompt_map.keys())[0]
        key_last =list(prompt_map.keys())[-1]

        def get_current_prompt_embeds_from_text(
                context: List[int] = None,
                video_length : int = 0
                ):
            center_frame = context[len(context)//2]

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

        ### image
        if self.ip_adapter:
            im_prompt_embeds_map = {}
            ip_im_map = ip_adapter_map["images"]

            ip_im_map = dict(sorted(ip_im_map.items()))

            ip_im_list = [ip_im_map[key_frame] for key_frame in ip_im_map.keys()]

            positive, negative = self.ip_adapter.get_image_embeds(ip_im_list)

            bs_embed, seq_len, _ = positive.shape
            positive = positive.repeat(1, 1, 1)
            positive = positive.view(bs_embed * 1, seq_len, -1)

            bs_embed, seq_len, _ = negative.shape
            negative = negative.repeat(1, 1, 1)
            negative = negative.view(bs_embed * 1, seq_len, -1)

            if do_classifier_free_guidance:
                negative = negative.chunk(negative.shape[0], 0)
                positive = positive.chunk(positive.shape[0], 0)
            else:
                positive = positive.chunk(positive.shape[0], 0)

            for i, key_frame in enumerate(ip_im_map):
                if do_classifier_free_guidance:
                    im_prompt_embeds_map[key_frame] = torch.cat([negative[i] , positive[i]])
                else:
                    im_prompt_embeds_map[key_frame] = positive[i]

            im_key_first =list(ip_im_map.keys())[0]
            im_key_last =list(ip_im_map.keys())[-1]

        def get_current_prompt_embeds_from_image(
                context: List[int] = None,
                video_length : int = 0
                ):
            center_frame = context[len(context)//2]

            key_prev = im_key_last
            key_next = im_key_first

            for p in ip_im_map.keys():
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
                return im_prompt_embeds_map[key_prev]

            rate = dist_prev / (dist_prev + dist_next)

            return im_prompt_embeds_map[key_prev] * (1-rate) + im_prompt_embeds_map[key_next] * (rate)


        def get_current_prompt_embeds(
                context: List[int] = None,
                video_length : int = 0
                ):
            text_emb = get_current_prompt_embeds_from_text(context, video_length)
            if self.ip_adapter:
                image_emb = get_current_prompt_embeds_from_image(context, video_length)
                return torch.cat([text_emb,image_emb], dim=1)
            else:
                return text_emb


        # 3.5 Prepare controlnet variables

        if self.controlnet_map:
            if controlnet_max_models_on_vram < len(self.controlnet_map):
                for _, type_str in zip( range(controlnet_max_models_on_vram-1) ,self.controlnet_map):
                    self.controlnet_map[type_str].to(device=device)
            else:
                for type_str in self.controlnet_map:
                    self.controlnet_map[type_str].to(device=device)


        # controlnet_image_map
        # { 0 : { "type_str" : IMAGE, "type_str2" : IMAGE }  }
        # { "type_str" : { 0 : IMAGE, 15 : IMAGE }  }
        controlnet_image_map= None

        if controlnet_image_map_org:
            controlnet_image_map= {key: {} for key in controlnet_type_map}
            for key_frame_no in controlnet_image_map_org:
                for t, img in controlnet_image_map_org[key_frame_no].items():
                    controlnet_image_map[t][key_frame_no] = self.prepare_image(
                                                        image=img,
                                                        width=width,
                                                        height=height,
                                                        batch_size=1 * 1,
                                                        num_images_per_prompt=1,
                                                        device=device,
                                                        dtype=self.controlnet_map[t].dtype,
                                                        do_classifier_free_guidance=do_classifier_free_guidance,
                                                        guess_mode=False,
                                                    )

        # { "0_type_str" : { "scales" = [0.1, 0.3, 0.5, 1.0, 0.5, 0.3, 0.1], "frames"=[125, 126, 127, 0, 1, 2, 3] }}
        controlnet_scale_map = {}
        controlnet_affected_list = np.zeros(video_length,dtype = int)

        if controlnet_image_map:
            for type_str in controlnet_image_map:
                for key_frame_no in controlnet_image_map[type_str]:
                    scale_list = controlnet_type_map[type_str]["control_scale_list"]
                    scale_list = scale_list[0: context_frames]
                    scale_len = len(scale_list)

                    if controlnet_is_loop:
                        frames = [ i%video_length for i in range(key_frame_no-scale_len, key_frame_no+scale_len+1)]

                        controlnet_scale_map[str(key_frame_no) + "_" + type_str] = {
                            "scales" : scale_list[::-1] + [1.0] + scale_list,
                            "frames" : frames,
                        }
                    else:
                        frames = [ i for i in range(max(0, key_frame_no-scale_len), min(key_frame_no+scale_len+1, video_length))]

                        controlnet_scale_map[str(key_frame_no) + "_" + type_str] = {
                            "scales" : scale_list[:key_frame_no][::-1] + [1.0] + scale_list[:video_length-key_frame_no-1],
                            "frames" : frames,
                        }

                    controlnet_affected_list[frames] = 1

        def controlnet_is_affected( frame_index:int):
            return controlnet_affected_list[frame_index]

        def get_controlnet_scale(
                type: str,
                cur_step: int,
                step_length: int,
                ):
            s = controlnet_type_map[type]["control_guidance_start"]
            e = controlnet_type_map[type]["control_guidance_end"]
            keep = 1.0 - float(cur_step / len(timesteps) < s or (cur_step + 1) / step_length > e)

            scale = controlnet_type_map[type]["controlnet_conditioning_scale"]

            return keep * scale

        def get_controlnet_variable(
                type_str: str,
                cur_step: int,
                step_length: int,
                target_frames: List[int],
                ):
            cont_vars = []

            if not controlnet_image_map:
                return None

            if type_str not in controlnet_image_map:
                return None

            for fr, img in controlnet_image_map[type_str].items():

                if fr in target_frames:
                    cont_vars.append( {
                        "frame_no" : fr,
                        "image" : img,
                        "cond_scale" : get_controlnet_scale(type_str, cur_step, step_length),
                        "guess_mode" : controlnet_type_map[type_str]["guess_mode"]
                    } )

            return cont_vars

        # 3.9. Preprocess reference image
        c_ref_enable = controlnet_ref_map is not None

        if c_ref_enable:
            ref_image = controlnet_ref_map["ref_image"]

            ref_image = self.prepare_ref_image(
                image=ref_image,
                width=width,
                height=height,
                batch_size=1 * 1,
                num_images_per_prompt=1,
                device=device,
                dtype=prompt_embeds.dtype,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=latents_device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            prompt_embeds.dtype,
            latents_device,  # keep latents on cpu for sequential mode
            generator,
            latents,
        )

        # 5.9. Prepare reference latent variables
        if c_ref_enable:
            ref_image_latents = self.prepare_ref_latents(
                ref_image,
                context_frames * 1,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )
            ref_image_latents = rearrange(ref_image_latents, "(b f) c h w -> b c f h w", f=context_frames)

            # 5.99. Modify self attention and group norm
            self.prepare_controlnet_ref_only(
                ref_image_latents=ref_image_latents,
                batch_size=context_frames,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                attention_auto_machine_weight=controlnet_ref_map["attention_auto_machine_weight"],
                gn_auto_machine_weight=controlnet_ref_map["gn_auto_machine_weight"],
                style_fidelity=controlnet_ref_map["style_fidelity"],
                reference_attn=controlnet_ref_map["reference_attn"],
                reference_adain=controlnet_ref_map["reference_adain"],
                _scale_pattern=controlnet_ref_map["scale_pattern"],
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.5 - Infinite context loop shenanigans
        context_scheduler = get_context_scheduler(context_schedule)
        total_steps = get_total_steps(
            context_scheduler,
            timesteps,
            num_inference_steps,
            latents.shape[2],
            context_frames,
            context_stride,
            context_overlap,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=total_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                stopwatch_start()

                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
                )

                # { "0_type_str" : (down_samples, mid_sample)  }
                controlnet_result={}

                def get_controlnet_result(context: List[int] = None):
                    #logger.info(f"get_controlnet_result called {context=}")

                    if controlnet_image_map is None:
                        return None, None

                    hit = False
                    for n in context:
                        if controlnet_is_affected(n):
                            hit=True
                            break
                    if hit == False:
                        return None, None

                    _down_block_res_samples=[]

                    first_down = list(list(controlnet_result.values())[0].values())[0][0]
                    first_mid = list(list(controlnet_result.values())[0].values())[0][1]
                    for ii in range(len(first_down)):
                        _down_block_res_samples.append(
                            torch.zeros(
                                (first_down[ii].shape[0], first_down[ii].shape[1], len(context) ,*first_down[ii].shape[3:]),
                                device=device,
                                dtype=first_down[ii].dtype,
                                ))
                    _mid_block_res_samples =  torch.zeros(
                                    (first_mid.shape[0], first_mid.shape[1], len(context) ,*first_mid.shape[3:]),
                                    device=device,
                                    dtype=first_mid.dtype,
                                    )

                    for fr in controlnet_result:
                        for type_str in controlnet_result[fr]:
                            result = str(fr) + "_" + type_str

                            val = controlnet_result[fr][type_str]
                            cur_down = [ v.to(device = device, dtype=first_down[0].dtype) for v in val[0] ]
                            cur_mid =val[1].to(device = device, dtype=first_mid.dtype)
                            loc =  list(set(context) & set(controlnet_scale_map[result]["frames"]))
                            scales = []

                            for o in loc:
                                for j, f in enumerate(controlnet_scale_map[result]["frames"]):
                                    if o == f:
                                        scales.append(controlnet_scale_map[result]["scales"][j])
                                        break
                            loc_index=[]

                            for o in loc:
                                for j, f in enumerate( context ):
                                    if o==f:
                                        loc_index.append(j)
                                        break

                            mod = torch.tensor(scales).to(device, dtype=cur_mid.dtype)

                            add = cur_mid * mod[None,None,:,None,None]
                            _mid_block_res_samples[:, :, loc_index, :, :] = _mid_block_res_samples[:, :, loc_index, :, :] + add

                            for ii in range(len(cur_down)):
                                add = cur_down[ii] * mod[None,None,:,None,None]
                                _down_block_res_samples[ii][:, :, loc_index, :, :] = _down_block_res_samples[ii][:, :, loc_index, :, :] + add

                    return _down_block_res_samples, _mid_block_res_samples

                def process_controlnet( target_frames: List[int] = None ):
                    #logger.info(f"process_controlnet called {target_frames=}")
                    nonlocal controlnet_result

                    controlnet_samples_on_vram = 0

                    loc =  list(set(target_frames) & set(controlnet_result.keys()))

                    controlnet_result = {key: controlnet_result[key] for key in loc}

                    target_frames = list(set(target_frames) - set(loc))

                    def sample_to_device( sample ):
                        nonlocal controlnet_samples_on_vram

                        if controlnet_max_samples_on_vram <= controlnet_samples_on_vram:
                            down_samples = [ v.to(device = torch.device("cpu")) for v in sample[0] ]
                            mid_sample = sample[1].to(device = torch.device("cpu"))
                        else:
                            if sample[0][0].device != device:
                                down_samples = [ v.to(device = device) for v in sample[0] ]
                                mid_sample = sample[1].to(device = device)
                            else:
                                down_samples = sample[0]
                                mid_sample = sample[1]
                            controlnet_samples_on_vram += 1
                        return down_samples, mid_sample


                    for fr in controlnet_result:
                        for type_str in controlnet_result[fr]:
                            controlnet_result[fr][type_str] = sample_to_device(controlnet_result[fr][type_str])

                    for type_str in controlnet_type_map:
                        cont_vars = get_controlnet_variable(type_str, i, len(timesteps), target_frames)
                        if not cont_vars:
                            continue

                        org_device = self.controlnet_map[type_str].device
                        if org_device != device:
                            self.controlnet_map[type_str] = self.controlnet_map[type_str].to(device=device)

                        for cont_var in cont_vars:
                            frame_no = cont_var["frame_no"]

                            latent_model_input = (
                                latents[:, :, [frame_no]]
                                .to(device)
                                .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                            )
                            control_model_input = self.scheduler.scale_model_input(latent_model_input, t)[:, :, 0]
                            controlnet_prompt_embeds = get_current_prompt_embeds([frame_no], latents.shape[2])

                            down_samples, mid_sample = self.controlnet_map[type_str](
                                control_model_input,
                                t,
                                encoder_hidden_states=controlnet_prompt_embeds,
                                controlnet_cond=cont_var["image"],
                                conditioning_scale=cont_var["cond_scale"],
                                guess_mode=cont_var["guess_mode"],
                                return_dict=False,
                            )

                            for ii in range(len(down_samples)):
                                down_samples[ii] = rearrange(down_samples[ii], "(b f) c h w -> b c f h w", f=1)
                            mid_sample = rearrange(mid_sample, "(b f) c h w -> b c f h w", f=1)

                            if frame_no not in controlnet_result:
                                controlnet_result[frame_no] = {}

                            controlnet_result[frame_no][type_str] = sample_to_device((down_samples, mid_sample))

                        if org_device != device:
                            self.controlnet_map[type_str] = self.controlnet_map[type_str].to(device=org_device)

                #logger.info(f"STEP start")
                stopwatch_record("STEP start")

                for context in context_scheduler(
                    i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
                ):
                    if controlnet_image_map:
                        controlnet_target = list(range(context[0]-context_frames, context[0])) + context + list(range(context[-1]+1, context[-1]+1+context_frames))
                        controlnet_target = [f%video_length for f in controlnet_target]
                        controlnet_target = list(set(controlnet_target))

                        process_controlnet(controlnet_target)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents[:, :, context]
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    cur_prompt = get_current_prompt_embeds(context, latents.shape[2])

                    down_block_res_samples,mid_block_res_sample = get_controlnet_result(context)

                    if c_ref_enable:
                        # ref only part
                        noise = randn_tensor(
                            ref_image_latents.shape, generator=generator, device=device, dtype=ref_image_latents.dtype
                        )

                        ref_xt = self.scheduler.add_noise(
                            ref_image_latents,
                            noise,
                            t.reshape(
                                1,
                            ),
                        )
                        ref_xt = self.scheduler.scale_model_input(ref_xt, t)

                        stopwatch_record("C_REF_MODE write start")

                        C_REF_MODE = "write"
                        self.unet(
                            ref_xt,
                            t,
                            encoder_hidden_states=cur_prompt,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )

                        stopwatch_record("C_REF_MODE write end")

                        C_REF_MODE = "read"

                    # predict the noise residual

                    stopwatch_record("normal unet start")
                    pred = self.unet(
                        latent_model_input.to(self.unet.device, self.unet.dtype),
                        t,
                        encoder_hidden_states=cur_prompt,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    stopwatch_record("normal unet end")

                    pred = pred.to(dtype=latents.dtype, device=latents.device)
                    noise_pred[:, :, context] = noise_pred[:, :, context] + pred
                    counter[:, :, context] = counter[:, :, context] + 1
                    progress_bar.update()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents.to(latents_device),
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                stopwatch_stop("LOOP end")

        if c_ref_enable:
            self.unload_controlnet_ref_only(
                reference_attn=controlnet_ref_map["reference_attn"],
                reference_adain=controlnet_ref_map["reference_adain"],
            )

        if self.ip_adapter:
            show_gpu("before unload ip_adapter")
            self.ip_adapter.unload()
            self.ip_adapter = None
            torch.cuda.empty_cache()
            show_gpu("after unload ip_adapter")

        # Return latents if requested (this will never be a dict)
        if not output_type == "latent":
            video = self.decode_latents(latents)
        else:
            video = latents

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def freeze(self):
        logger.debug("Freezing pipeline...")
        _ = self.unet.eval()
        self.unet = self.unet.requires_grad_(False)
        self.unet.train = nop_train

        _ = self.text_encoder.eval()
        self.text_encoder = self.text_encoder.requires_grad_(False)
        self.text_encoder.train = nop_train

        _ = self.vae.eval()
        self.vae = self.vae.requires_grad_(False)
        self.vae.train = nop_train
