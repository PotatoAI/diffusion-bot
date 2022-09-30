import torch
import os
import uuid
import coloredlogs
import functools
import asyncio
import traceback
from logging import info, error, warn
from diffusers import StableDiffusionPipeline, ModelMixin
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from concurrent.futures import ProcessPoolExecutor

coloredlogs.install(level='INFO')

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
        warn('Skipping NSFW check')
        return images, [False]


class DiffusionThing:
    def __init__(self):
        info('Initializing pipeline')
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
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


diffusion = DiffusionThing()


async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text.replace("/gen", "").strip()
    # await update.message.reply_text('👍')
    try:
        fname = await diffusion.run(prompt)
        with open(fname, 'rb') as photo:
            await update.message.reply_photo(photo=photo, caption=prompt)
    except Exception as e:
        error(e)
        traceback.print_exc()
        await update.message.reply_text(f"Exception: {e}")
    # os.remove(fname)


if __name__ == '__main__':
    token = os.getenv('TG_TOKEN')
    application = ApplicationBuilder().token(token).build()
    start_handler = CommandHandler('gen', generate)
    application.add_handler(start_handler)
    application.run_polling()
