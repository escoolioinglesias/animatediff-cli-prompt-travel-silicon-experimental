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
from tqdm.rich import tqdm

from animatediff import __version__, get_dir
from animatediff.settings import ModelConfig, get_model_config
from animatediff.utils.tagger import get_labels
from animatediff.utils.util import (extract_frames, get_resized_image,
                                    path_from_cwd, prepare_groundingDINO,
                                    prepare_propainter, prepare_sam_hq,
                                    prepare_softsplat)

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
    offset: Annotated[
        int,
        typer.Option(
            "--offset",
            "-of",
            min=0,
            max=3600,
            help="offset in seconds. '-d 30 -of 1200' means to use 1200-1230 seconds of the input video",
        ),
    ] = 0,
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
    size_of_short_edge: Annotated[
        int,
        typer.Option(
            "--short-edge",
            "-sh",
            min=100,
            max=1024,
            help="size of short edge",
        ),
    ] = 512,
    predicte_interval: Annotated[
        int,
        typer.Option(
            "--predicte-interval",
            "-p",
            min=1,
            max=120,
            help="Interval of frames to be predicted",
        ),
    ] = 1,
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
    without_confidence: Annotated[
        bool,
        typer.Option(
            "--no-confidence-format",
            "-ncf",
            is_flag=True,
            help="confidence token format or not. ex. '(close-up:0.57), (monochrome:1.1)' -> 'close-up, monochrome'",
        ),
    ] = False,
    is_no_danbooru_format: Annotated[
        bool,
        typer.Option(
            "--no-danbooru-format",
            "-ndf",
            is_flag=True,
            help="danbooru token format or not. ex. 'bandaid_on_leg, short_hair' -> 'bandaid on leg, short hair'",
        ),
    ] = False,
):
    """Create a config file for video stylization"""
    is_danbooru_format = not is_no_danbooru_format
    with_confidence = not without_confidence
    logger.info(f"{org_movie=}")
    logger.info(f"{config_org=}")
    logger.info(f"{ignore_list=}")
    logger.info(f"{out_dir=}")
    logger.info(f"{fps=}")
    logger.info(f"{duration=}")
    logger.info(f"{offset=}")
    logger.info(f"{aspect_ratio=}")
    logger.info(f"{size_of_short_edge=}")
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

    extract_frames(org_movie, fps, controlnet_img_dir.joinpath("controlnet_tile"), aspect_ratio, duration, offset, size_of_short_edge)

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
        is_danbooru_format=is_danbooru_format,
        is_cpu = False,
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

    for m in model_config.controlnet_map:
        if isinstance(model_config.controlnet_map[m] ,dict):
            if "control_scale_list" in model_config.controlnet_map[m]:
                model_config.controlnet_map[m]["control_scale_list"]=[]

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
        "original_video":{
            "path":org_movie,
            "aspect_ratio":aspect_ratio,
            "offset":offset,
        },
        "create_mask": [
            "person"
        ],
        "composite": {
            "fg_list": [
                {
                    "path" : " absolute path to frame dir ",
                    "mask_path" : " absolute path to mask dir (this is optional) ",
                    "mask_prompt" : "person"
                },
                {
                    "path" : " absolute path to frame dir ",
                    "mask_path" : " absolute path to mask dir (this is optional) ",
                    "mask_prompt" : "cat"
                },
            ],
            "bg_frame_dir": "Absolute path to the BG frame directory",
            "hint": ""
        },
        "0":{
            "width": width,
            "height": height,
            "length": length,
            "context": 16,
            "overlap": 16//4,
            "stride": 0,
        },
        "1":{
            "steps": model_config.steps,
            "guidance_scale": model_config.guidance_scale,
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
    frame_offset: Annotated[
        int,
        typer.Option(
            "--frame-offset",
            "-FO",
            min=0,
            max=999999,
            help="Frame offset at generation.",
            rich_help_panel="Generation",
        ),
    ] = 0,
):
    """Run video stylization"""
    from animatediff.cli import generate

    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)

    if length > 0:
        model_config.stylize_config["0"]["length"] = min(model_config.stylize_config["0"]["length"] - frame_offset, length)
        if "1" in model_config.stylize_config:
            model_config.stylize_config["1"]["length"] = min(model_config.stylize_config["1"]["length"] - frame_offset, length)

    if frame_offset > 0:
        org_controlnet_img_dir = data_dir.joinpath( model_config.controlnet_map["input_image_dir"] )
        new_controlnet_img_dir = org_controlnet_img_dir.parent / "00_tmp_controlnet_image"
        if new_controlnet_img_dir.is_dir():
            shutil.rmtree(new_controlnet_img_dir)
        new_controlnet_img_dir.mkdir(parents=True, exist_ok=True)

        for c in ["controlnet_canny","controlnet_depth","controlnet_inpaint","controlnet_ip2p","controlnet_lineart","controlnet_lineart_anime","controlnet_mlsd","controlnet_normalbae","controlnet_openpose","controlnet_scribble","controlnet_seg","controlnet_shuffle","controlnet_softedge","controlnet_tile"]:
            src_dir = org_controlnet_img_dir.joinpath(c)
            dst_dir = new_controlnet_img_dir.joinpath(c)
            if src_dir.is_dir():
                dst_dir.mkdir(parents=True, exist_ok=True)

                frame_length = model_config.stylize_config["0"]["length"]

                src_imgs = sorted(glob.glob( os.path.join(src_dir, "[0-9]*.png"), recursive=False))
                for img in src_imgs:
                    n = int(Path(img).stem)
                    if n in range(frame_offset, frame_offset + frame_length):
                        dst_img_path = dst_dir.joinpath( f"{n-frame_offset:08d}.png" )
                        shutil.copy(img, dst_img_path)

        new_prompt_map = {}
        for p in model_config.prompt_map:
            n = int(p)
            if n in range(frame_offset, frame_offset + frame_length):
                new_prompt_map[str(n-frame_offset)]=model_config.prompt_map[p]

        model_config.prompt_map = new_prompt_map

        model_config.controlnet_map["input_image_dir"] = os.path.relpath(new_controlnet_img_dir.absolute(), data_dir)

        tmp_config_path = stylize_dir.joinpath("prompt_tmp.json")
        tmp_config_path.write_text(model_config.json(indent=4), encoding="utf-8")
        config_org = tmp_config_path


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
        if rife_img_dir.is_dir():
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

    model_config.steps = model_config.stylize_config["1"]["steps"] if "steps" in model_config.stylize_config["1"] else model_config.steps
    model_config.guidance_scale = model_config.stylize_config["1"]["guidance_scale"] if "guidance_scale" in model_config.stylize_config["1"] else model_config.guidance_scale

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




@stylize.command(no_args_is_help=True)
def interpolate(
    frame_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to frame dir"),
    ] = ...,
    interpolation_multiplier: Annotated[
        int,
        typer.Option(
            "--interpolation_multiplier",
            "-m",
            min=1,
            max=10,
            help="interpolation_multiplier",
        ),
    ] = 1,
):
    """Interpolation with original frames. This function does not work well if the shape of the subject is changed from the original video. Large movements can also ruin the picture.(Since this command is experimental, it is better to use other interpolation methods in most cases.)"""

    try:
        import cupy
    except:
        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.info(f"cupy is required to run interpolate")
        logger.info(f"Your CUDA version is {torch.version.cuda}")
        logger.info(f"Please find the installation method of cupy for your CUDA version from the following URL")
        logger.info(f"https://docs.cupy.dev/en/latest/install.html#installing-cupy-from-pypi")
        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    prepare_softsplat()

    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_org = frame_dir.parent.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)

    if "original_video" in model_config.stylize_config:
        org_video = Path(model_config.stylize_config["original_video"]["path"])
        offset = model_config.stylize_config["original_video"]["offset"]
        aspect_ratio = model_config.stylize_config["original_video"]["aspect_ratio"]
    else:
        logger.warn('!!! The following parameters are required !!!')
        logger.warn('"stylize_config": {')
        logger.warn('    "original_video": {')
        logger.warn('        "path": "C:\\my_movie\\test.mp4",')
        logger.warn('        "aspect_ratio": 0.6666,')
        logger.warn('        "offset": 0')
        logger.warn('    },')
        raise ValueError('model_config.stylize_config["original_video"] not found')


    save_dir = frame_dir.parent.joinpath(f"optflow_{time_str}")

    org_frame_dir = save_dir.joinpath("org_frame")
    org_frame_dir.mkdir(parents=True, exist_ok=True)

    stylize_frame = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))
    stylize_frame_num = len(stylize_frame)

    duration = int(stylize_frame_num / model_config.output["fps"]) + 1

    extract_frames(org_video, model_config.output["fps"] * interpolation_multiplier, org_frame_dir,aspect_ratio,duration,offset)

    W, H = Image.open(stylize_frame[0]).size

    org_frame = sorted(glob.glob( os.path.join(org_frame_dir, "[0-9]*.png"), recursive=False))

    for org in tqdm(org_frame):
        img = get_resized_image(org, W, H)
        img.save(org)

    output_dir = save_dir.joinpath("warp_img")
    output_dir.mkdir(parents=True, exist_ok=True)

    from animatediff.softmax_splatting.run import estimate2

    for sty1,sty2 in tqdm(zip(stylize_frame,stylize_frame[1:]), total=len(stylize_frame[1:])):
        sty1 = Path(sty1)
        sty2 = Path(sty2)

        head = int(sty1.stem)

        sty1_img = Image.open(sty1)
        sty2_img = Image.open(sty2)

        guide_frames=[org_frame_dir.joinpath(f"{g:08d}.png") for g in range(head*interpolation_multiplier, (head+1)*interpolation_multiplier)]

        guide_frames=[Image.open(g) for g in guide_frames]

        result = estimate2(sty1_img, sty2_img, guide_frames, "data/models/softsplat/softsplat-lf")

        shutil.copy( frame_dir.joinpath(f"{head:08d}.png"), output_dir.joinpath(f"{head*interpolation_multiplier:08d}.png"))

        offset = head*interpolation_multiplier + 1
        for i, r in enumerate(result):
            r.save( output_dir.joinpath(f"{offset+i:08d}.png") )


    from animatediff.generate import save_output


    frames = sorted(glob.glob( os.path.join(output_dir, "[0-9]*.png"), recursive=False))
    out_images = []
    for f in frames:
        out_images.append(Image.open(f))

    model_config.output["fps"] *= interpolation_multiplier

    out_file = save_dir.joinpath(f"01_{model_config.output['fps']}fps")
    save_output(out_images,output_dir,out_file,model_config.output,True,save_frames=None,save_video=None)

    out_file = save_dir.joinpath(f"00_original")
    save_output(out_images,org_frame_dir,out_file,model_config.output,True,save_frames=None,save_video=None)


@stylize.command(no_args_is_help=True)
def create_mask(
    stylize_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to stylize dir"),
    ] = ...,
    frame_dir: Annotated[
        Path,
        typer.Option(
            "--frame_dir",
            "-f",
            path_type=Path,
            file_okay=False,
            help="Path to source frames directory. default is 'STYLIZE_DIR/00_controlnet_image/controlnet_tile'",
        ),
    ] = None,
    box_threshold: Annotated[
        float,
        typer.Option(
            "--box_threshold",
            "-b",
            min=0.0,
            max=1.0,
            help="box_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.3,
    text_threshold: Annotated[
        float,
        typer.Option(
            "--text_threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="text_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.25,
    mask_padding: Annotated[
        int,
        typer.Option(
            "--mask_padding",
            "-p",
            min=-100,
            max=100,
            help="padding pixel value",
            rich_help_panel="create mask",
        ),
    ] = 0,
    low_vram: Annotated[
        bool,
        typer.Option(
            "--low_vram",
            "-lo",
            is_flag=True,
            help="low vram mode",
            rich_help_panel="create mask/tag",
        ),
    ] = False,
    ignore_list: Annotated[
        Path,
        typer.Option(
            "--ignore-list",
            "-g",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="path to ignore token list file",
            rich_help_panel="create tag",
        ),
    ] = Path("config/prompts/ignore_tokens.txt"),
    predicte_interval: Annotated[
        int,
        typer.Option(
            "--predicte-interval",
            "-p",
            min=1,
            max=120,
            help="Interval of frames to be predicted",
            rich_help_panel="create tag",
        ),
    ] = 1,
    general_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-th",
            min=0.0,
            max=1.0,
            help="threshold for general token confidence",
            rich_help_panel="create tag",
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
            rich_help_panel="create tag",
        ),
    ] = 0.85,
    without_confidence: Annotated[
        bool,
        typer.Option(
            "--no-confidence-format",
            "-ncf",
            is_flag=True,
            help="confidence token format or not. ex. '(close-up:0.57), (monochrome:1.1)' -> 'close-up, monochrome'",
            rich_help_panel="create tag",
        ),
    ] = False,
    is_no_danbooru_format: Annotated[
        bool,
        typer.Option(
            "--no-danbooru-format",
            "-ndf",
            is_flag=True,
            help="danbooru token format or not. ex. 'bandaid_on_leg, short_hair' -> 'bandaid on leg, short hair'",
            rich_help_panel="create tag",
        ),
    ] = False,
):
    """Create mask from prompt"""
    from animatediff.utils.mask import create_bg, create_fg

    is_danbooru_format = not is_no_danbooru_format
    with_confidence = not without_confidence

    prepare_sam_hq(low_vram)
    prepare_groundingDINO()
    prepare_propainter()

    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)

    if frame_dir is None:
        frame_dir = stylize_dir / "00_controlnet_image/controlnet_tile"

    if not frame_dir.is_dir():
        raise ValueError(f'{frame_dir=} does not exist.')


    create_mask_list = []
    if "create_mask" in model_config.stylize_config:
        create_mask_list = model_config.stylize_config["create_mask"]
    else:
        raise ValueError('model_config.stylize_config["create_mask"] not found')

    output_list = []

    frame_len = len(sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False)))

    masked_area = [None for f in range(frame_len)]


    def create_controlnet_dir(controlnet_root):
        for c in ["controlnet_canny","controlnet_depth","controlnet_inpaint","controlnet_ip2p","controlnet_lineart","controlnet_lineart_anime","controlnet_mlsd","controlnet_normalbae","controlnet_openpose","controlnet_scribble","controlnet_seg","controlnet_shuffle","controlnet_softedge","controlnet_tile"]:
            c_dir = controlnet_root.joinpath(c)
            c_dir.mkdir(parents=True, exist_ok=True)

    for i,mask_token in enumerate(create_mask_list):
        fg_dir = stylize_dir.joinpath(f"fg_{i:02d}_{time_str}")
        fg_dir.mkdir(parents=True, exist_ok=True)

        create_controlnet_dir( fg_dir / "00_controlnet_image" )

        fg_masked_dir = fg_dir / "00_controlnet_image/controlnet_tile"

        fg_mask_dir = fg_dir / "00_mask"
        fg_mask_dir.mkdir(parents=True, exist_ok=True)

        masked_area = create_fg(
            mask_token=mask_token,
            frame_dir=frame_dir,
            output_dir=fg_masked_dir,
            output_mask_dir=fg_mask_dir,
            masked_area_list=masked_area,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            mask_padding=mask_padding,
            sam_checkpoint= "data/models/SAM/sam_hq_vit_l.pth" if not low_vram else "data/models/SAM/sam_hq_vit_b.pth",
        )

        logger.info(f"mask from [{mask_token}] are output to {fg_dir}")

        shutil.copytree(fg_masked_dir, fg_masked_dir.parent.joinpath("controlnet_ip2p"), dirs_exist_ok=True)

        output_list.append(fg_dir)

    torch.cuda.empty_cache()

    bg_dir = stylize_dir.joinpath(f"bg_{time_str}")
    bg_dir.mkdir(parents=True, exist_ok=True)
    create_controlnet_dir( bg_dir / "00_controlnet_image" )
    bg_inpaint_dir = bg_dir / "00_controlnet_image/controlnet_tile"

    create_bg(frame_dir, bg_inpaint_dir, masked_area,
              use_half = True,
              raft_iter = 20,
              subvideo_length=80 if not low_vram else 50,
              neighbor_length=10 if not low_vram else 8,
              ref_stride=10 if not low_vram else 8,
              low_vram = low_vram,
              )

    logger.info(f"background are output to {bg_dir}")

    shutil.copytree(bg_inpaint_dir, bg_inpaint_dir.parent.joinpath("controlnet_ip2p"), dirs_exist_ok=True)

    output_list.append(bg_dir)

    torch.cuda.empty_cache()

    black_list = []
    if ignore_list.is_file():
        with open(ignore_list) as f:
            black_list = [s.strip() for s in f.readlines()]

    for output in output_list:

        model_config.prompt_map = get_labels(
            frame_dir= output / "00_controlnet_image/controlnet_tile",
            interval=predicte_interval,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            ignore_tokens=black_list,
            with_confidence=with_confidence,
            is_danbooru_format=is_danbooru_format,
            is_cpu = False,
        )

        model_config.controlnet_map["input_image_dir"] = os.path.relpath((output / "00_controlnet_image" ).absolute(), data_dir)

        save_config_path = output.joinpath("prompt.json")
        save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")




@stylize.command(no_args_is_help=True)
def composite(
    stylize_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to stylize dir"),
    ] = ...,
    box_threshold: Annotated[
        float,
        typer.Option(
            "--box_threshold",
            "-b",
            min=0.0,
            max=1.0,
            help="box_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.3,
    text_threshold: Annotated[
        float,
        typer.Option(
            "--text_threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="text_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.25,
    mask_padding: Annotated[
        int,
        typer.Option(
            "--mask_padding",
            "-p",
            min=-100,
            max=100,
            help="padding pixel value",
            rich_help_panel="create mask",
        ),
    ] = 0,
    low_vram: Annotated[
        bool,
        typer.Option(
            "--low_vram",
            "-lo",
            is_flag=True,
            help="low vram mode",
            rich_help_panel="create mask/tag",
        ),
    ] = False,
):
    """composite FG and BG"""

    from animatediff.utils.composite import composite
    from animatediff.utils.mask import create_fg, loda_mask_list

    prepare_sam_hq(low_vram)

    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)


    composite_config = {}
    if "composite" in model_config.stylize_config:
        composite_config = model_config.stylize_config["composite"]
    else:
        raise ValueError('model_config.stylize_config["composite"] not found')

    save_dir = stylize_dir.joinpath(f"cp_{time_str}")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")


    bg_dir = composite_config["bg_frame_dir"]
    bg_dir = Path(bg_dir)
    if not bg_dir.is_dir():
        raise ValueError('model_config.stylize_config["composite"]["bg_frame_dir"] not valid')

    frame_len = len(sorted(glob.glob( os.path.join(bg_dir, "[0-9]*.png"), recursive=False)))

    fg_list = composite_config["fg_list"]

    for i, fg_param in enumerate(fg_list):
        mask_token = fg_param["mask_prompt"]
        frame_dir = Path(fg_param["path"])
        if not frame_dir.is_dir():
            logger.warn(f"{frame_dir=} not valid -> skip")
            continue

        mask_dir = Path(fg_param["mask_path"])
        if not mask_dir.is_dir():
            logger.info(f"{mask_dir=} not valid -> create mask")

            fg_tmp_dir = save_dir.joinpath(f"fg_{i:02d}_{time_str}")
            fg_tmp_dir.mkdir(parents=True, exist_ok=True)

            masked_area_list = [None for f in range(frame_len)]

            mask_list = create_fg(
                mask_token=mask_token,
                frame_dir=frame_dir,
                output_dir=fg_tmp_dir,
                output_mask_dir=None,
                masked_area_list=masked_area_list,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                mask_padding=mask_padding,
                sam_checkpoint= "data/models/SAM/sam_hq_vit_l.pth" if not low_vram else "data/models/SAM/sam_hq_vit_b.pth",
            )
        else:
            logger.info(f"use {mask_dir=} as mask")

            masked_area_list = [None for f in range(frame_len)]

            mask_list = loda_mask_list(mask_dir, masked_area_list, mask_padding)

        output_dir = save_dir.joinpath(f"bg_{i:02d}_{time_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

        composite(bg_dir, frame_dir, output_dir, mask_list)

        bg_dir = output_dir


    from animatediff.generate import save_output

    frames = sorted(glob.glob( os.path.join(bg_dir, "[0-9]*.png"), recursive=False))
    out_images = []
    for f in frames:
        out_images.append(Image.open(f))

    out_file = save_dir.joinpath(f"composite")
    save_output(out_images,bg_dir,out_file,model_config.output,True,save_frames=None,save_video=None)

    logger.info(f"output to {out_file}")
