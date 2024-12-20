import torch
from diffusers import UNet2DModel
import tqdm

class DDPM:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start,beta_end,num_train_timesteps,dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim = 0)
        self.timesteps = torch.arange(num_train_timesteps-1,-1,-1)

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor
    ):
        """x_t = sqrt(alphas_cumprod_xt)*x0 + sqrt(1 - alphas_cumprod_xt)*eps"""
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device ,dtype=original_samples.dtype)
        noise = noise.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

         # \sqrt{\bar\alpha_t}
        sqrt_alpha_prod = alphas_cumprod[timesteps].flatten() ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # \sqrt{1 - \bar\alpha_t}
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]).flatten() ** 0.5
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    @torch.no_grad()
    def sample(
            self,
            unet:UNet2DModel,
            batch_size:int,
            in_channels:int,
            sample_size:int,
    ):
        """迭代求解images = mean + variance*epsilon"""
        betas = self.betas.to(unet.device)
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alphas_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        images = torch.randn((batch_size, in_channels, sample_size, sample_size), device=unet.device)

        for timestep in tqdm(timesteps, desc='Sampling'):
            pred_noise:torch.Tensor = unet(images,timestep).sample

             # mean of q(x_{t-1}|x_t)
            alpha_t = alphas[timestep]
            alpha_cumprod_t = alphas_cumprod[timestep]
            sqrt_alpha_t = alpha_t ** 0.5
            one_minus_alpha_t = 1.0 - alpha_t
            sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t) ** 0.5
            mean = (images - one_minus_alpha_t / sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_t

            # variance of q(x_{t-1}|x_t)
            if timestep > 0:
                beta_t = betas[timestep]
                one_minus_alpha_cumprod_t_minus_one = 1.0 - alphas_cumprod[timestep - 1]
                one_divided_by_sigma_square = alpha_t / beta_t + 1.0 / one_minus_alpha_cumprod_t_minus_one
                variance = (1.0 / one_divided_by_sigma_square) ** 0.5
            else:
                variance = torch.zeros_like(timestep)

            epsilon = torch.randn_like(images)
            images = mean + variance * epsilon
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images

    

        