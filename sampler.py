import torch
import utils
import torch.nn.functional as F

class Sampler():
    def __init__(self, model, timesteps, start, end, img_size):
        self.model = model
        self.start = start
        self.end = end
        self.img_size = img_size
        self.T = timesteps
        self.betas = self.schedule_beta()
        self.alphas = 1.0 - self.betas
        self.alpha_cprod = torch.cumprod(self.alphas, 0)
        self.alpha_cprod_prev = F.pad(self.alpha_cprod[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_cprod = torch.sqrt(self.alpha_cprod)
        self.one_minus_sqrt_alpha_cprod = torch.sqrt(1 - self.alpha_cprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alpha_cprod)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cprod_prev) / (1.0 - self.alpha_cprod)

    def schedule_beta(self):
        return torch.linspace(self.start, self.end, self.T)
    
    def forward_diffusion(self, x_0, t, device):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = utils.get_values_by_index(self.sqrt_alpha_cprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = utils.get_values_by_index(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def sample_timestep(self, x, t, model):
        with torch.no_grad():
            betas_t = utils.get_values_by_index(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = utils.get_values_by_index(
                self.one_minus_sqrt_alpha_cprod, t, x.shape
            )
            sqrt_recip_alphas_t = utils.get_values_by_index(self.sqrt_recip_alphas, t, x.shape)
            
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
            )
            posterior_variance_t = utils.get_values_by_index(self.posterior_variance, t, x.shape)

            if t == 0:
                return model_mean
            else:
                noise = torch.randn_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise 
          
    def sample_image(self, device):
        with torch.no_grad():
            img = torch.randn((1, 1, self.img_size, self.img_size), device=device)

            for t in range(0, self.T)[::-1]:
                t = torch.full((1, ), t, device=device, dtype=torch.long)
                img = self.sample_timestep(img, t, self.model)
            img = torch.clamp(img, -1.0, 1.0)

            return img