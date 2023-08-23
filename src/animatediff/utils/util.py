from os import PathLike
from pathlib import Path
from typing import List

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image
from tqdm.rich import tqdm


def save_frames(video: Tensor, frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = rearrange(video, "b c t h w -> t b c h w")
    for idx, frame in enumerate(tqdm(frames, desc=f"Saving frames to {frames_dir.stem}")):
        save_image(frame, frames_dir.joinpath(f"{idx:04d}.png"))

def save_imgs(imgs:List[Image.Image], frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(tqdm(imgs, desc=f"Saving frames to {frames_dir.stem}")):
        img.save( frames_dir.joinpath(f"{idx:04d}.png") )

def save_video(video: Tensor, save_path: PathLike, fps: int = 8):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim == 5:
        # batch, channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(0, 2, 1, 3, 4).squeeze(0)
    elif video.ndim == 4:
        # channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"video must be 4 or 5 dimensional, got {video.ndim}")

    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    frames = frames.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        fp=save_path, format="GIF", append_images=images[1:], save_all=True, duration=(1 / fps * 1000), loop=0
    )


def path_from_cwd(path: PathLike) -> str:
    path = Path(path)
    return str(path.absolute().relative_to(Path.cwd()))


def resize_for_condition_image(input_image: Image, us_width: int, us_height: int):
    input_image = input_image.convert("RGB")
    H = int(round(us_height / 64.0)) * 64
    W = int(round(us_width / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def get_resized_images(org_images_path: List[str], us_width: int, us_height: int):

    images = [Image.open( p ) for p in org_images_path]

    W, H = images[0].size

    if us_width == -1:
        us_width = W/H * us_height
    elif us_height == -1:
        us_height = H/W * us_width

    return [resize_for_condition_image(img, us_width, us_height) for img in images]

def get_resized_image(org_image_path: str, us_width: int, us_height: int):

    image = Image.open( org_image_path )

    W, H = image.size

    if us_width == -1:
        us_width = W/H * us_height
    elif us_height == -1:
        us_height = H/W * us_width

    return resize_for_condition_image(image, us_width, us_height)

