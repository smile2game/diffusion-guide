import torch
from tqdm import tqdm
from diffusers import UNet2DModel
from .ddpm import DDPM


model = UNet2DModel.from_pretrained('ddpm-animefaces-64').cuda()
ddpm = DDPM()
images = ddpm.sample(model, 32, 3, 64)

from diffusers.utils import make_image_grid, numpy_to_pil
image_grid = make_image_grid(numpy_to_pil(images), rows=4, cols=8)
image_grid.save('ddpm-sample-results.png')