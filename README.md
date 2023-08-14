# AnimateDiff prompt travel

AnimateDiff with prompt travel

I added a experimental feature to animatediff-cli to change the prompt in the middle of the frame.

It seems to work surprisingly well!

### Example
- standing -> walking -> spider webs:2.0 -> sitting
- Left : output of "animatediff generate -c config/prompts/prompt_travel.json -W 512 -H 768 -L128 -C 16"
- Right : output of "animatediff tile-upscale PATH_TO_TARGET_FRAME_DIRECTORY -c config/prompts/prompt_travel.json -W 512"


### Installation(for windows)
Same as the original animatediff-cli
```sh
git clone https://github.com/s9roll7/animatediff-cli.git
cd animatediff-cli
py -3.10 -m venv venv
venv\Scripts\activate.bat
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
python -m pip install xformers
```
(https://www.reddit.com/r/StableDiffusion/comments/157c0wl/working_animatediff_cli_windows_install/)


### How To Use
Almost same as the original animatediff-cli, but with a slight change in config format.
```json
# prompt_travel.json
{
  "name": "sample",
  "path": "share/Stable-diffusion/mistoonAnime_v20.safetensors",  # Specify Checkpoint as a path relative to /animatediff-cli/data
  "motion_module": "models/motion-module/mm_sd_v14.ckpt",         # Specify motion module as a path relative to /animatediff-cli/data
  "compile": false,
  "seed": [
    341774366206100         # -1 means random
  ],
  "scheduler": "ddim",
  "steps": 40,
  "guidance_scale": 20,     # cfg scale
  "clip_skip": 2,
  "prompt_map": {           # "FRAME" : "PROMPT" format
    "0":  "masterpiece, best quality, a beautiful and detailed portriat of muffet, monster girl,((purple body:1.3)),humanoid, arachnid, anthro,((fangs)),pigtails,hair bows,5 eyes,spider girl,6 arms,solo,smile standing, clothed, open mouth, awesome and detailed background, holding teapot, holding teacup, 6 hands,detailed hands,((spider webs:1.0)), storefront that sells pastries and tea,bloomers,(red and black clothing),inside,pouring into teacup,muffetwear",
    "32":  "masterpiece, best quality, a beautiful and detailed portriat of muffet, monster girl,((purple body:1.3)),humanoid, arachnid, anthro,((fangs)),pigtails,hair bows,5 eyes,spider girl,6 arms,solo,(((walking))), clothed, open mouth, awesome and detailed background, holding teapot, holding teacup, 6 hands,detailed hands,((spider webs:1.0)), storefront that sells pastries and tea,bloomers,(red and black clothing),inside,pouring into teacup,muffetwear",
    "64":  "masterpiece, best quality, a beautiful and detailed portriat of muffet, monster girl,((purple body:1.3)),humanoid, arachnid, anthro,((fangs)),pigtails,hair bows,5 eyes,spider girl,6 arms,solo,(((running))), clothed, open mouth, awesome and detailed background, holding teapot, holding teacup, 6 hands,detailed hands,((spider webs:2.0)), storefront that sells pastries and tea,bloomers,(red and black clothing),inside,pouring into teacup,muffetwear,wide angle lens, fish eye effect",
    "96":  "masterpiece, best quality, a beautiful and detailed portriat of muffet, monster girl,((purple body:1.3)),humanoid, arachnid, anthro,((fangs)),pigtails,hair bows,5 eyes,spider girl,6 arms,solo,(((sitting))), clothed, open mouth, awesome and detailed background, holding teapot, holding teacup, 6 hands,detailed hands,((spider webs:1.0)), storefront that sells pastries and tea,bloomers,(red and black clothing),inside,pouring into teacup,muffetwear"
  },
  "n_prompt": [
    "(worst quality, low quality:1.4),nudity,simple background,border,mouth closed,text, patreon,bed,bedroom,white background,((monochrome)),sketch,(pink body:1.4),7 arms,8 arms,4 arms"
  ],
  "lora_map": {             # "PATH_TO_LORA" : STRENGTH format
    "share/Lora/muffet_v2.safetensors" : 1.0,                     # Specify lora as a path relative to /animatediff-cli/data
    "share/Lora/add_detail.safetensors" : 1.0                     # Lora support is limited. Not all formats can be used!!!
  },
  "upscale_config": {       # config for tile-upscale
    "scheduler": "ddim",
    "steps": 20,
    "strength": 0.5,
    "guidance_scale": 10
  }
}
```

```sh
cd animatediff-cli
venv\Scripts\activate.bat

# with this setup, it took about a minute to generate in my environment(RTX4090). VRAM usage was 6-7 GB
# width 256 / height 384 / length 128 frames / context 16 frames
animatediff generate -c config/prompts/prompt_travel.json -W 256 -H 384 -L128 -C 16
# 5min / 9-10GB
animatediff generate -c config/prompts/prompt_travel.json -W 512 -H 768 -L128 -C 16

# upscale using controlnet tile
# specify the directory of the frame generated in the above step
# here, width=512 is specified, but even if the original size is 512, it is effective in increasing detail
animatediff tile-upscale PATH_TO_TARGET_FRAME_DIRECTORY -c config/prompts/prompt_travel.json -W 512

```

### Limitations
- lora support is limited. Not all formats can be used!!!
- It is not possible to specify lora in the prompt.


Below is the original readme.

----------------------------------------------------------


# animatediff
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/neggles/animatediff-cli/main.svg)](https://results.pre-commit.ci/latest/github/neggles/animatediff-cli/main)

animatediff refactor, ~~because I can.~~ with significantly lower VRAM usage.

Also, **infinite generation length support!** yay!

# LoRA loading is ABSOLUTELY NOT IMPLEMENTED YET!

This can theoretically run on CPU, but it's not recommended. Should work fine on a GPU, nVidia or otherwise,
but I haven't tested on non-CUDA hardware. Uses PyTorch 2.0 Scaled-Dot-Product Attention (aka builtin xformers)
by default, but you can pass `--xformers` to force using xformers if you *really* want.

### How To Use

1. Lie down
2. Try not to cry
3. Cry a lot

### but for real?

Okay, fine. But it's still a little complicated and there's no webUI yet.

```sh
git clone https://github.com/neggles/animatediff-cli
cd animatediff-cli
python3.10 -m venv .venv
source .venv/bin/activate
# install Torch. Use whatever your favourite torch version >= 2.0.0 is, but, good luck on non-nVidia...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install the rest of all the things (probably! I may have missed some deps.)
python -m pip install -e '.[dev]'
# you should now be able to
animatediff --help
# There's a nice pretty help screen with a bunch of info that'll print here.
```

From here you'll need to put whatever checkpoint you want to use into `data/models/sd`, copy
one of the prompt configs in `config/prompts`, edit it with your choices of prompt and model (model
paths in prompt .json files are **relative to `data/`**, e.g. `models/sd/vanilla.safetensors`), and
off you go.

Then it's something like (for an 8GB card):
```sh
animatediff generate -c 'config/prompts/waifu.json' -W 576 -H 576 -L 128 -C 16
```
You may have to drop `-C` down to 8 on cards with less than 8GB VRAM, and you can raise it to 20-24
on cards with more. 24 is max.

N.B. generating 128 frames is _**slow...**_

## RiFE!

I have added experimental support for [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)
using the `animatediff rife interpolate` command. It has fairly self-explanatory help, and it has
been tested on Linux, but I've **no idea** if it'll work on Windows.

Either way, you'll need ffmpeg installed on your system and present in PATH, and you'll need to
download the rife-ncnn-vulkan release for your OS of choice from the GitHub repo (above). Unzip it, and
place the extracted folder at `data/rife/`. You should have a `data/rife/rife-ncnn-vulkan` executable, or `data\rife\rife-ncnn-vulkan.exe` on Windows.

You'll also need to reinstall the repo/package with:
```py
python -m pip install -e '.[rife]'
```
or just install `ffmpeg-python` manually yourself.

Default is to multiply each frame by 8, turning an 8fps animation into a 64fps one, then encode
that to a 60fps WebM. (If you pick GIF mode, it'll be 50fps, because GIFs are cursed and encode
frame durations as 1/100ths of a second).

Seems to work pretty well...

## TODO:

In no particular order:

- [x] Infinite generation length support
- [x] RIFE support for motion interpolation (`rife-ncnn-vulkan` isn't the greatest implementation)
- [x] Export RIFE interpolated frames to a video file (webm, mp4, animated webp, hevc mp4, gif, etc.)
- [x] Generate infinite length animations on a 6-8GB card (at 512x512 with 8-frame context, but hey it'll do)
- [x] Torch SDP Attention (makes xformers optional)
- [x] Support for `clip_skip` in prompt config
- [x] Experimental support for `torch.compile()` (upstream Diffusers bugs slow this down a little but it's still zippy)
- [x] Batch your generations with `--repeat`! (e.g. `--repeat 10` will repeat all your prompts 10 times)
- [x] Call the `animatediff.cli.generate()` function from another Python program without reloading the model every time
- [x] Drag remaining old Diffusers code up to latest (mostly)
- [ ] Add a webUI (maybe, there are people wrapping this already so maybe not?)
- [ ] img2img support (start from an existing image and continue)
- [ ] Stop using custom modules where possible (should be able to use Diffusers for almost all of it)
- [ ] Automatic generate-then-interpolate-with-RIFE mode

## Credits:

see [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) (very little of this is my work)

n.b. the copyright notice in `COPYING` is missing the original authors' names, solely because
the original repo (as of this writing) has no name attached to the license. I have, however,
used the same license they did (Apache 2.0).
