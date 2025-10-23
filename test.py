import torch
from diffusers import AutoPipelineForText2Image

torch.set_num_threads(1)

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("Apple").images[0]
image.save("test.png")
