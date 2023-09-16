import logging
from os import PathLike
from pathlib import Path
from typing import List

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)

def save_frames(video: Tensor, frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = rearrange(video, "b c t h w -> t b c h w")
    for idx, frame in enumerate(tqdm(frames, desc=f"Saving frames to {frames_dir.stem}")):
        save_image(frame, frames_dir.joinpath(f"{idx:08d}.png"))

def save_imgs(imgs:List[Image.Image], frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(tqdm(imgs, desc=f"Saving frames to {frames_dir.stem}")):
        img.save( frames_dir.joinpath(f"{idx:08d}.png") )

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

def get_resized_image2(org_image_path: str, size: int):

    image = Image.open( org_image_path )

    W, H = image.size

    if size < 0:
        return resize_for_condition_image(image, W, H)

    if W < H:
        us_width = size
        us_height = int(size * H/W)
    else:
        us_width = int(size * W/H)
        us_height = size

    return resize_for_condition_image(image, us_width, us_height)


def show_gpu(comment):
    pass
#    import GPUtil
#    torch.cuda.synchronize()
#    logger.info(comment)
#    GPUtil.showUtilization()


PROFILE_ON = False

def start_profile():
    if PROFILE_ON:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        return pr
    else:
        return None

def end_profile(pr, file_name):
    if PROFILE_ON:
        import io
        import pstats

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open(file_name, 'w+') as f:
            f.write(s.getvalue())

STOPWATCH_ON = False

time_record = []
start_time = 0

def stopwatch_start():
    global start_time,time_record
    import time

    if STOPWATCH_ON:
        time_record = []
        torch.cuda.synchronize()
        start_time = time.time()

def stopwatch_record(comment):
    import time

    if STOPWATCH_ON:
        torch.cuda.synchronize()
        time_record.append(((time.time() - start_time) , comment))

def stopwatch_stop(comment):

    if STOPWATCH_ON:
        stopwatch_record(comment)

        for rec in time_record:
            logger.info(rec)


def prepare_ip_adapter():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/ip_adapter/models/image_encoder", exist_ok=True)
    for hub_file in [
        "models/image_encoder/config.json",
        "models/image_encoder/pytorch_model.bin",
        "models/ip-adapter-plus_sd15.bin",
        "models/ip-adapter_sd15.bin",
        "models/ip-adapter-plus-face_sd15.bin"
    ]:
        path = Path(hub_file)

        saved_path = "data/models/ip_adapter" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="h94/IP-Adapter", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/ip_adapter"
        )

def prepare_motion_module():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/motion-module", exist_ok=True)
    for hub_file in [
        "mm_sd_v15_v2.ckpt",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/motion-module" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="guoyww/animatediff", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/motion-module"
        )

def prepare_wd14tagger():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/WD14tagger", exist_ok=True)
    for hub_file in [
        "model.onnx",
        "selected_tags.csv",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/WD14tagger" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/WD14tagger"
        )



def extract_frames(movie_file_path, fps, out_dir, aspect_ratio, duration):
    import ffmpeg

    probe = ffmpeg.probe(movie_file_path)
    video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video['width'])
    height = int(video['height'])

    node = ffmpeg.input( str(movie_file_path.resolve()) )

    node = node.filter( "fps", fps=fps )

    if duration > 0:
        node = node.trim(duration=duration)

    if aspect_ratio > 0:
        # aspect ratio (width / height)
        ww = round(height * aspect_ratio)
        if ww < width:
            x= (width - ww)//2
            y= 0
            w = ww
            h = height
        else:
            hh = round(width/aspect_ratio)
            x = 0
            y = (height - hh)//2
            w = width
            h = hh
        logger.info(f"crop to {w=},{h=}")
        node = node.crop(x, y, w, h)

    node = node.output( str(out_dir.resolve().joinpath("%08d.png")), start_number=0 )

    node.run(quiet=True, overwrite_output=True)



def is_v2_motion_module(motion_module_path:Path):
    return (motion_module_path.stem in ["mm_sd_v15_v2"] )     # TODO Decent identification method



tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation

def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()
