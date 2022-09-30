from diffusers import StableDiffusionPipeline, ModelMixin


class NoCheck(ModelMixin):
    """Can be used in place of safety checker. Use responsibly and at your own risk."""

    def __init__(self):
        super().__init__()
        self.register_parameter(name='asdf',
                                param=torch.nn.Parameter(torch.randn(3)))

    def forward(self, images=None, **kwargs):
        return images, [False]


class DiffusionThing:

    def __init__(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", use_auth_token=True)
        pipe.safety_checker = NoCheck()
        pipe.enable_attention_slicing()
        self.pipe = pipe.to("mps")

    def run(self):
        prompt = "a photo of an astronaut riding a horse on mars"
        image = self.pipe(prompt).images[0]
        image.save('output.png')


DiffusionThing().run()
