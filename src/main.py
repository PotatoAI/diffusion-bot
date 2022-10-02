import torch
import os
import uuid
import coloredlogs
import functools
import asyncio
import traceback
import subprocess
from logging import info, error, warning
from diffusers import StableDiffusionPipeline, ModelMixin
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
from concurrent.futures import ProcessPoolExecutor

coloredlogs.install(level='INFO')

model_id = os.getenv('MODEL', 'CompVis/stable-diffusion-v1-4')
info(f"Using {model_id} model")

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
        warning('Skipping NSFW check')
        return images, [False]


class Diffuser:

    def __init__(self):
        info('Initializing pipeline')
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
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

    @run_in_executor
    def run(self, prompt: str) -> str:
        info(f'Generating "{prompt}"')
        with torch.autocast(self.device):
            result = self.pipe([prompt], num_inference_steps=50)
        image = result.images[0]
        id = uuid.uuid1()
        fname = f"images/{id}.png"
        info(f'Saving to "{fname}"')
        image.save(fname)
        return fname


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
    prompt = update.message.text.replace("/gen", "").strip()
    try:
        if len(prompt) > 0:
            await update.message.reply_text(f'👍 Generating "{prompt}"')
            fname = await diffuser.run(prompt)
            with open(fname, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=prompt)
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
