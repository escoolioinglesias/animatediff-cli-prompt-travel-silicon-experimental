import glob
import logging
import os.path
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from PIL import Image

from animatediff import __version__, get_dir
from animatediff.settings import ModelConfig, get_model_config
from animatediff.utils.tagger import get_labels
from animatediff.utils.util import extract_frames, path_from_cwd

logger = logging.getLogger(__name__)



stylize: typer.Typer = typer.Typer(
    name="stylize",
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    help="stylize video",
)

data_dir = get_dir("data")


@stylize.command(no_args_is_help=True)
def create_config(
    org_movie: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=True, dir_okay=False, exists=True, help="Path to movie file"),
    ] = ...,
    config_org: Annotated[
        Path,
        typer.Option(
            "--config-org",
            "-c",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="Path to original config file",
        ),
    ] = Path("config/prompts/prompt_travel.json"),
    ignore_list: Annotated[
        Path,
        typer.Option(
            "--ignore-list",
            "-g",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="path to ignore token list file",
        ),
    ] = Path("config/prompts/ignore_tokens.txt"),
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            "-o",
            path_type=Path,
            file_okay=False,
            help="output directory",
        ),
    ] = Path("stylize/"),
    fps: Annotated[
        int,
        typer.Option(
            "--fps",
            "-f",
            min=1,
            max=120,
            help="fps",
        ),
    ] = 8,
    duration: Annotated[
        int,
        typer.Option(
            "--duration",
            "-d",
            min=-1,
            max=3600,
            help="Video duration in seconds. -1 means that the duration of the input video is used as is",
        ),
    ] = -1,
    aspect_ratio: Annotated[
        float,
        typer.Option(
            "--aspect-ratio",
            "-a",
            min=-1,
            max=5.0,
            help="aspect ratio (width / height). (ex. 512 / 512 = 1.0 , 512 / 768 = 0.6666 , 768 / 512 = 1.5) -1 means that the aspect ratio of the input video is used as is.",
        ),
    ] = -1,
    predicte_interval: Annotated[
        int,
        typer.Option(
            "--predicte-interval",
            "-p",
            min=1,
            max=120,
            help="Interval of frames to be predicted",
        ),
    ] = 16,
    general_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-th",
            min=0.0,
            max=1.0,
            help="threshold for general token confidence",
        ),
    ] = 0.35,
    character_threshold: Annotated[
        float,
        typer.Option(
            "--threshold2",
            "-th2",
            min=0.0,
            max=1.0,
            help="threshold for character token confidence",
        ),
    ] = 0.85,
    with_confidence: Annotated[
        bool,
        typer.Option(
            "--confidence-format",
            "-cf",
            is_flag=True,
            help="confidence token format or not",
        ),
    ] = True,
    is_danbooru_format: Annotated[
        bool,
        typer.Option(
            "--danbooru-format",
            "-df",
            is_flag=True,
            help="danbooru token format or not",
        ),
    ] = True,

):
    logger.info(f"{org_movie=}")
    logger.info(f"{config_org=}")
    logger.info(f"{ignore_list=}")
    logger.info(f"{out_dir=}")
    logger.info(f"{fps=}")
    logger.info(f"{duration=}")
    logger.info(f"{aspect_ratio=}")
    logger.info(f"{predicte_interval=}")
    logger.info(f"{general_threshold=}")
    logger.info(f"{character_threshold=}")
    logger.info(f"{with_confidence=}")
    logger.info(f"{is_danbooru_format=}")

    model_config: ModelConfig = get_model_config(config_org)

    # get a timestamp for the output directory
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # make the output directory
    save_dir = out_dir.joinpath(f"{time_str}-{model_config.save_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Will save outputs to ./{path_from_cwd(save_dir)}")

    controlnet_img_dir = save_dir.joinpath("00_controlnet_image")

    for c in ["controlnet_canny","controlnet_depth","controlnet_inpaint","controlnet_ip2p","controlnet_lineart","controlnet_lineart_anime","controlnet_mlsd","controlnet_normalbae","controlnet_openpose","controlnet_scribble","controlnet_seg","controlnet_shuffle","controlnet_softedge","controlnet_tile"]:
        c_dir = controlnet_img_dir.joinpath(c)
        c_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(org_movie, fps, controlnet_img_dir.joinpath("controlnet_tile"), aspect_ratio, duration)

    shutil.copytree(controlnet_img_dir.joinpath("controlnet_tile"), controlnet_img_dir.joinpath("controlnet_ip2p"), dirs_exist_ok=True)

    black_list = []
    if ignore_list.is_file():
        with open(ignore_list) as f:
            black_list = [s.strip() for s in f.readlines()]

    model_config.prompt_map = get_labels(
        frame_dir=controlnet_img_dir.joinpath("controlnet_tile"),
        interval=predicte_interval,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        ignore_tokens=black_list,
        with_confidence=with_confidence,
        is_danbooru_format=is_danbooru_format
    )


    model_config.head_prompt = ""
    model_config.tail_prompt = ""
    model_config.controlnet_map["input_image_dir"] = os.path.relpath(controlnet_img_dir.absolute(), data_dir)
    model_config.controlnet_map["is_loop"] = False

    model_config.controlnet_map["controlnet_tile"] = {
      "enable": True,
      "use_preprocessor":True,
      "guess_mode":False,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list":[]
    }
    model_config.controlnet_map["controlnet_ip2p"] = {
      "enable": True,
      "use_preprocessor":True,
      "guess_mode":False,
      "controlnet_conditioning_scale": 0.5,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list":[]
    }

    model_config.output = {
        "format" : "mp4",
        "fps" : fps,
        "encode_param":{
            "crf": 10
        }
    }

    img = Image.open( controlnet_img_dir.joinpath("controlnet_tile/00000000.png") )
    W, H = img.size

    if W < H:
        width = 512
        height = int(512 * H/W)
    else:
        width = int(512 * W/H)
        height = 512

    width = int(width//8*8)
    height = int(height//8*8)

    length = len(glob.glob( os.path.join(controlnet_img_dir.joinpath("controlnet_tile"), "[0-9]*.png"), recursive=False))

    model_config.stylize_config={
        "0":{
            "width": width,
            "height": height,
            "length": length,
            "context": 16,
            "overlap": 16//4,
            "stride": 0,
        },
        "1":{
            "width": int(width * 1.5 //8*8),
            "height": int(height * 1.5 //8*8),
            "length": length,
            "context": 8,
            "overlap": 8//4,
            "stride": 0,
            "controlnet_tile":{
                "enable": True,
                "use_preprocessor":True,
                "guess_mode":False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list":[]
            },
            "controlnet_ip2p": {
                "enable": False,
                "use_preprocessor":True,
                "guess_mode":False,
                "controlnet_conditioning_scale": 0.5,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list":[]
            },
            "ip_adapter": False,
            "reference": False,
            "interpolation_multiplier": 1
        }
    }

    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    logger.info(f"config = { save_config_path }")
    logger.info(f"stylize_dir = { save_dir }")

    logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info(f"Hint. Edit the config file before starting the generation")
    logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info(f"1. Change 'path' and 'motion_module' as needed")
    logger.info(f"2. Enter the 'head_prompt' or 'tail_prompt' with your preferred prompt, quality prompt, lora trigger word, or any other prompt you wish to add.")
    logger.info(f"3. Change 'n_prompt' as needed")
    logger.info(f"4. Add the lora you need to 'lora_map'")
    logger.info(f"5. If you do not like the default settings, edit 'ip_adapter_map' or 'controlnet_map'. \nIf you want to change the controlnet type, you need to replace the input image.")
    logger.info(f"6. Change 'stylize_config' as needed. By default, it is generated twice: once for normal generation and once for upscaling.\nIf you don't need upscaling, delete the whole '1'.")
    logger.info(f"7. Change 'output' as needed. Changing the 'fps' at this timing is not recommended as it will change the playback speed.\nIf you want to change the fps, specify it with the create-config option")


@stylize.command(no_args_is_help=True)
def generate(
    stylize_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to stylize dir"),
    ] = ...,
    length: Annotated[
        int,
        typer.Option(
            "--length",
            "-L",
            min=-1,
            max=9999,
            help="Number of frames to generate. -1 means that the value in the config file is referenced.",
            rich_help_panel="Generation",
        ),
    ] = -1,
):
    from animatediff.cli import generate

    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)

    if length > 0:
        model_config.stylize_config["0"]["length"] = min(model_config.stylize_config["0"]["length"], length)
        if "1" in model_config.stylize_config:
            model_config.stylize_config["1"]["length"] = min(model_config.stylize_config["1"]["length"], length)

    output_0_dir = generate(
        config_path=config_org,
        width=model_config.stylize_config["0"]["width"],
        height=model_config.stylize_config["0"]["height"],
        length=model_config.stylize_config["0"]["length"],
        context=model_config.stylize_config["0"]["context"],
        overlap=model_config.stylize_config["0"]["overlap"],
        stride=model_config.stylize_config["0"]["stride"],
        out_dir=stylize_dir
    )

    torch.cuda.empty_cache()

    output_0_dir = output_0_dir.rename(output_0_dir.parent / f"{time_str}_{0:02d}")


    if "1" not in model_config.stylize_config:
        logger.info(f"Stylized results are output to {output_0_dir}")
        return

    logger.info(f"Intermediate files have been output to {output_0_dir}")

    output_0_img_dir = glob.glob( os.path.join(output_0_dir, "00-[0-9]*"), recursive=False)[0]

    interpolation_multiplier = 1
    if "interpolation_multiplier" in model_config.stylize_config["1"]:
        interpolation_multiplier = model_config.stylize_config["1"]["interpolation_multiplier"]

    if interpolation_multiplier > 1:
        from animatediff.rife.rife import rife_interpolate

        rife_img_dir = stylize_dir.joinpath(f"{1:02d}_rife_frame")
        shutil.rmtree(rife_img_dir)
        rife_img_dir.mkdir(parents=True, exist_ok=True)

        rife_interpolate(output_0_img_dir, rife_img_dir, interpolation_multiplier)
        model_config.stylize_config["1"]["length"] *= interpolation_multiplier

        if model_config.output:
            model_config.output["fps"] *= interpolation_multiplier
        if model_config.prompt_map:
            model_config.prompt_map = { str(int(i)*interpolation_multiplier): model_config.prompt_map[i] for i in model_config.prompt_map }

        output_0_img_dir = rife_img_dir


    controlnet_img_dir = stylize_dir.joinpath("01_controlnet_image")

    for c in ["controlnet_canny","controlnet_depth","controlnet_inpaint","controlnet_ip2p","controlnet_lineart","controlnet_lineart_anime","controlnet_mlsd","controlnet_normalbae","controlnet_openpose","controlnet_scribble","controlnet_seg","controlnet_shuffle","controlnet_softedge","controlnet_tile"]:
        c_dir = controlnet_img_dir.joinpath(c)
        c_dir.mkdir(parents=True, exist_ok=True)


    ip2p_for_upscale = model_config.stylize_config["1"]["controlnet_ip2p"]["enable"]
    ip_adapter_for_upscale = model_config.stylize_config["1"]["ip_adapter"]
    ref_for_upscale = model_config.stylize_config["1"]["reference"]

    shutil.copytree(output_0_img_dir, controlnet_img_dir.joinpath("controlnet_tile"), dirs_exist_ok=True)
    if ip2p_for_upscale:
        shutil.copytree(controlnet_img_dir.joinpath("controlnet_tile"), controlnet_img_dir.joinpath("controlnet_ip2p"), dirs_exist_ok=True)


    model_config.controlnet_map["input_image_dir"] = os.path.relpath(controlnet_img_dir.absolute(), data_dir)

    model_config.controlnet_map["controlnet_tile"] = model_config.stylize_config["1"]["controlnet_tile"]
    model_config.controlnet_map["controlnet_ip2p"] = model_config.stylize_config["1"]["controlnet_ip2p"]

    if "controlnet_ref" in model_config.controlnet_map:
        model_config.controlnet_map["controlnet_ref"]["enable"] = ref_for_upscale

    model_config.ip_adapter_map["enable"] = ip_adapter_for_upscale

    save_config_path = stylize_dir.joinpath("prompt_01.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    output_1_dir = generate(
        config_path=save_config_path,
        width=model_config.stylize_config["1"]["width"],
        height=model_config.stylize_config["1"]["height"],
        length=model_config.stylize_config["1"]["length"],
        context=model_config.stylize_config["1"]["context"],
        overlap=model_config.stylize_config["1"]["overlap"],
        stride=model_config.stylize_config["1"]["stride"],
        out_dir=stylize_dir
    )

    output_1_dir = output_1_dir.rename(output_1_dir.parent / f"{time_str}_{1:02d}")

    logger.info(f"Stylized results are output to {output_1_dir}")
