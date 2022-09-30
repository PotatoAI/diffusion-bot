from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
pipe = pipe.to("mps")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save('output.png')
