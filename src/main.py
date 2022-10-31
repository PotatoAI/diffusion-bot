import torch
import os
import uuid
import coloredlogs
import functools
import asyncio
import traceback
import subprocess
from typing import Optional, List
from logging import info, error, warning, debug
from diffusers import StableDiffusionPipeline, ModelMixin
from telegram import Update, InputMediaPhoto
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
from concurrent.futures import ProcessPoolExecutor
from bot.args import GenerateArgs, gen_seed
from telegram import InputMediaPhoto
from pydantic import BaseModel

coloredlogs.install(level='INFO')

model_path = os.getenv('MODEL', 'CompVis/stable-diffusion-v1-4')
model_id = os.getenv('MODEL_ID', 'unknown')
info(f"Using {model_path} model")

pool_executor = ProcessPoolExecutor(1)


def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))

    return inner


class NoCheck(ModelMixin):
    """Can be used in place of safety checker. Use responsibly and at your own risk."""
    def __init__(self):
        super().__init__()
        self.register_parameter(name='asdf',
                                param=torch.nn.Parameter(torch.randn(3)))

    def forward(self, images=None, **kwargs):
        debug('Skipping NSFW check')
        return images, [False]


class GeneratedMedia(BaseModel):
    path: str
    caption: str
    seed: int


class Diffuser:
    def __init__(self):
        self.initialized = False

    def init(self):
        if not self.initialized:
            info(f'Initializing pipeline from {model_path}')
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=True)
            pipe.safety_checker = NoCheck()

            self.device = 'cuda'
            if torch.backends.mps.is_available():
                self.device = 'mps'
            info(f'Moving model to {self.device}')
            self.pipe = pipe.to(self.device)

            info('Enabling attention slicing for smaller VRAM usage')
            self.pipe.enable_attention_slicing()

        self.initialized = True

    @run_in_executor
    def run(self, args: GenerateArgs) -> List[GeneratedMedia]:
        self.init()

        files = []
        info(f'Generating "{args.prompt}"')

        if args.seedwalk != 0:
            args.seed = gen_seed()
            info(
                f'Setting args.seed to {args.seed} since seedwalk is {args.seedwalk}'
            )

        for i in range(0, args.count):
            seed = gen_seed()
            if args.seed:
                seed = args.seed + args.seedwalk * i
            info(f"Setting seed to {seed}")
            generator = torch.Generator(self.device).manual_seed(seed)

            with torch.autocast(self.device):
                result = self.pipe([args.prompt],
                                   generator=generator,
                                   width=512,
                                   height=768,
                                   num_inference_steps=50)

            for image in result.images:
                id = uuid.uuid1()
                fname = f"images/{model_id}_{seed}_{id}.png"
                info(f'Saving to "{fname}"')
                image.save(fname)
                files.append(
                    GeneratedMedia(path=fname,
                                   seed=seed,
                                   caption=f'{args.prompt} seed={seed}'))

        return files


def sh(cmd: str):
    info(cmd)
    subprocess.run(["bash", "-c", cmd])


class Upscaler:
    def __init__(self):
        self.runtime_dir = "BSRGAN"
        self.input_file = "input.jpg"
        self.output_file = "output.jpg"

    @run_in_executor
    def run(self, path: str) -> str:
        sh(f"mv -f {path} {self.runtime_dir}/{self.input_file}")
        sh(f"cd {self.runtime_dir} && make upscale-jpg")
        return f"{self.runtime_dir}/{self.output_file}"


diffuser = Diffuser()
upscaler = Upscaler()


async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info(update)
    args = GenerateArgs.from_prompt(update.message.text)
    try:
        args.sanity_check()
        if len(args.prompt) > 0:
            await update.message.reply_text(f'👍 Generating "{args.prompt}"')
            gen_media = await diffuser.run(args)
            caption = f'{args.prompt} seed={args.seed}'
            photos = [
                InputMediaPhoto(media=open(media.path, 'rb'),
                                caption=media.caption) for media in gen_media
            ]

            await update.message.reply_media_group(media=photos)
    except Exception as e:
        error(e)
        traceback.print_exc()
        await update.message.reply_text(f"Exception: {e}")
    # os.remove(fname)


async def upscale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info(update)
    if len(update.message.photo) > 0:
        await update.message.reply_text('👍 upscaling')
        fid = update.message.photo[-1].file_id
        file_info = await context.bot.get_file(fid)
        fpath = await file_info.download()
        out_path = await upscaler.run(fpath)
        info(out_path)
        with open(out_path, 'rb') as photo:
            await update.message.reply_photo(photo=photo,
                                             caption="Upscaled photo")


help_message = """
Commands:
/gen prompt - generate image based on prompt
"""


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(help_message)


if __name__ == '__main__':
    token = os.getenv('TG_TOKEN')
    application = ApplicationBuilder().token(token).build()
    help_handler = CommandHandler('help', help)
    start_handler = CommandHandler('gen', generate)
    up_handler = MessageHandler(None, upscale)
    application.add_handler(help_handler)
    application.add_handler(start_handler)
    application.add_handler(up_handler)
    application.run_polling()
